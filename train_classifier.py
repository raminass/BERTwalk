from functools import partial
import torch
import torch.nn as nn
import time
from modeling.models import BinaryClassification
from modeling.data import TrainData, TestData, classifier_collate
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from utils.commons import *
import random
import json
import argparse

# random.seed(1984)
torch.manual_seed(1984)

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():

    parser = argparse.ArgumentParser(description="Tune the trainded BBERTwalk model on classification task.")
    parser.add_argument("--model_name", type=str, default="Oct06_19-46-44_s-001", help="Trained model name.")
    parser.add_argument("--data_path", type=str, default="inputs/labeled_paths.txt", help="Data file.")

    return parser.parse_args()


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def train_fun(model, dataloader, model_params):
    lr = model_params["learning_rate"]
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(model_params["epochs"]):
        epoch_loss = 0
        epoch_acc = 0
        epoch_start_time = time.time()
        model.pre_class.all_prop_emb()
        for b, batch in enumerate(dataloader):
            optim.zero_grad()
            input = batch["input"]
            labels = batch["labels"].to(torch.float).to(device)
            src_mask = batch["src_mask"]
            src_key_padding_mask = batch["src_key_padding_mask"]

            y_pred = model(input.to(device), src_mask.to(device), src_key_padding_mask.to(device))
            loss = criterion(y_pred, labels.unsqueeze(1))
            acc = binary_acc(y_pred, labels.unsqueeze(1))

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optim.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        elapsed = time.time() - epoch_start_time
        loss_dict = {}
        val_loss = epoch_loss / (len(dataloader))
        loss_dict["total"] = val_loss

        acc_dict = {}
        val_acc = epoch_acc / (len(dataloader))
        acc_dict["total"] = val_acc

        print("-" * 89)
        print(f"| epoch {1+epoch:3d} | time: {elapsed:.2f}s | " f"loss {val_loss:.5f} | Acc {val_acc:.5f}")
        print("-" * 89)
        # writer.add_scalars("Loss", loss_dict, epoch + 1)


if __name__ == "__main__":

    args = parse_args()
    # Loading pre-trained
    name = args.model_name  # pre-trained model name
    checkpoint = torch.load(f"artifacts/{name}_model.pt")
    tokenizer = checkpoint["tokenizer"]
    # Loading pathway data, last token is the label
    with open(args.data_path, "r") as f:
        data = json.load(f)
    X = np.array([text.rsplit(" ", 1)[0] for text in data])  # walks only
    y = np.array([int(text.split(" ")[-1]) for text in data])  # labels only
    # prepare for training
    for net in checkpoint["networks"]:
        net["node_tokens"] = torch.tensor(tokenizer.encode(net["node_sequence"][0]).ids).to(device)
        net.edge_index = net.edge_index.to(device)
        net.edge_weight = net.edge_weight.to(device)

    checkpoint["model_params"]["epochs"] = 1

    metrics = ["auc", "fpr", "tpr", "thresholds"]
    results = {"test": {m: [] for m in metrics}}
    auc_all = []

    # CV training
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for i, (train, test) in enumerate(skf.split(X, y)):
        # instantiate model
        class_model = BinaryClassification(checkpoint)
        class_model.to(device)
        # building data
        train_data = TrainData(X[train].tolist(), y[train].tolist(), tokenizer)
        test_data = TestData(X[test].tolist(), y[test].tolist(), tokenizer)
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=checkpoint["model_params"]["batch_size"],
            shuffle=True,
            collate_fn=partial(classifier_collate, tokenizer=tokenizer),
        )
        test_loader = DataLoader(
            dataset=test_data, batch_size=1, collate_fn=partial(classifier_collate, tokenizer=tokenizer)
        )
        # train model for current fold
        train_fun(class_model, train_loader, checkpoint["model_params"])
        # evaluta model for current fold
        y_pred_list = []
        y_test = []
        class_model.eval()
        with torch.no_grad():
            for batch in test_loader:
                input = batch["input"]
                labels = batch["labels"]
                src_mask = batch["src_mask"]
                src_key_padding_mask = batch["src_key_padding_mask"]
                y_pred = class_model(input.to(device), src_mask.to(device), src_key_padding_mask.to(device))
                y_test_pred = torch.sigmoid(y_pred)
                y_pred_list.append(y_test_pred.cpu().numpy())
                y_test.append(labels.numpy())

        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        y_test = [a.squeeze().tolist() for a in y_test]
        # calculating useful metrics
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_list, pos_label=1)
        results["test"]["fpr"].append(fpr.tolist())
        results["test"]["tpr"].append(tpr.tolist())
        results["test"]["thresholds"].append(thresholds.tolist())
        results["test"]["auc"].append(auc(fpr, tpr))
        auc_all.append(auc(fpr, tpr))

    with open("results.json", "w") as fp:
        json.dump(results, fp)
    