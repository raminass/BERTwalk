from functools import partial
import torch
import torch.nn as nn
import time
from modeling.models import BinaryClassification, MultiClassification, TrainedBert
from modeling.data import TrainData, TestData, classifier_collate
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from torch.utils.tensorboard import SummaryWriter
from utils.commons import *
import random
import json
import argparse
from sklearn.linear_model import LogisticRegression

# random.seed(1984)
torch.manual_seed(1984)

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():

    parser = argparse.ArgumentParser(description="Tune the trainded BBERTwalk model on classification task.")
    parser.add_argument("--model_name", type=str, default="Oct06_19-46-44_s-001", help="Trained model name.")
    parser.add_argument("--data_path", type=str, default="inputs/signed_paths.txt", help="Data file.")

    return parser.parse_args()


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def train_fun(bert, model, dataloader, model_params):
    lr = model_params["learning_rate"]
    model.train()
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(300):
        epoch_loss = 0
        epoch_acc = 0
        epoch_start_time = time.time()
        # bert.pre_class.all_prop_emb()
        for b, batch in enumerate(dataloader):
            optim.zero_grad()
            input = batch["input"]
            labels = batch["labels"].to(torch.float).to(device)
            src_mask = batch["src_mask"]
            src_key_padding_mask = batch["src_key_padding_mask"]

            y_pred = model(bert(input.to(device), src_mask.to(device), src_key_padding_mask.to(device)))
            loss = criterion(y_pred, labels.long())
            # acc = binary_acc(y_pred, labels.unsqueeze(1))

            _, predicted = torch.max(y_pred.data, 1)
            acc = (predicted == labels).sum()/labels.shape[0]


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

    from sklearn.preprocessing import LabelBinarizer
    label_binarizer = LabelBinarizer().fit(y)

    # prepare for training
    for net in checkpoint["networks"]:
        net["node_tokens"] = torch.tensor(tokenizer.encode(net["node_sequence"][0]).ids).to(device)
        net.edge_index = net.edge_index.to(device)
        net.edge_weight = net.edge_weight.to(device)

    checkpoint["model_params"]["epochs"] = 1

    metrics = ["auc", "fpr", "tpr", "thresholds", "macro_f1"]
    results = {"test": {m: [] for m in metrics}}
    auc_all = []

    # CV training
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for i, (train, test) in enumerate(skf.split(X, y)):
        # instantiate model
        bert_model = TrainedBert(checkpoint).to(device)
        class_model = MultiClassification(checkpoint).to(device)

        # RF clf
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier()

        # building data
        train_data = TrainData(X[train].tolist(), y[train].tolist(), tokenizer)
        test_data = TestData(X[test].tolist(), y[test].tolist(), tokenizer)
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=checkpoint["model_params"]["batch_size"],
            shuffle=True,
            collate_fn=partial(classifier_collate, tokenizer=tokenizer),
        )

        # train clf
        x_feat = []
        y_label = []
        for batch in train_loader:
            input = batch["input"]
            labels = batch["labels"]
            src_mask = batch["src_mask"]
            src_key_padding_mask = batch["src_key_padding_mask"]
            x_feat.append(bert_model(input.to(device), src_mask.to(device), src_key_padding_mask.to(device)).cpu().detach().numpy())
            y_label.append(labels.numpy())

        x_cat_train = np.concatenate(x_feat, axis=0)
        y_cat_train = np.concatenate(y_label, axis=0)
        clf.fit(x_cat_train, y_cat_train)

        test_loader = DataLoader(
            dataset=test_data, batch_size=1, collate_fn=partial(classifier_collate, tokenizer=tokenizer)
        )
        x_feat = []
        y_label = []
        for batch in test_loader:
            input = batch["input"]
            labels = batch["labels"]
            src_mask = batch["src_mask"]
            src_key_padding_mask = batch["src_key_padding_mask"]
            x_feat.append(bert_model(input.to(device), src_mask.to(device), src_key_padding_mask.to(device)).cpu().detach().numpy())
            y_label.append(labels.numpy())
        
        x_cat_test = np.concatenate(x_feat, axis=0)
        y_cat_test = np.concatenate(y_label, axis=0)
        y_pred_bertwalk = clf.predict(x_cat_test)
        y_onehot_test = label_binarizer.transform(y_cat_test)

        from scipy.sparse import csr_matrix
        macro_f1 = f1_score(y_cat_test, y_pred_bertwalk, average="macro")

        fpr = [0,0,0]
        tpr = [0,0,0]
        roc_auc = [0,0,0]
        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], clf.predict_proba(x_cat_test)[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr_grid = np.linspace(0.0, 1.0, 1000)
        mean_tpr = np.zeros_like(fpr_grid)
        for i in range(3):
            mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
        mean_tpr /= 3


        # # train model for current fold
        # train_fun(bert_model, class_model, train_loader, checkpoint["model_params"])
        # # evaluta model for current fold
        # y_pred_list = []
        # y_test = []
        # class_model.eval()
        # with torch.no_grad():
        #     for batch in test_loader:
        #         input = batch["input"]
        #         labels = batch["labels"]
        #         src_mask = batch["src_mask"]
        #         src_key_padding_mask = batch["src_key_padding_mask"]
        #         y_pred = class_model(bert_model(input.to(device), src_mask.to(device), src_key_padding_mask.to(device)))
        #         y_test_pred = nn.functional.softmax(y_pred,dim=1)
        #         y_pred_list.append(y_test_pred.cpu().numpy())

        #         y_test.append(label_binarizer.transform(labels))

        # y_pred_list = np.array([a.squeeze().tolist() for a in y_pred_list])
        # y_test = np.array([a.squeeze().tolist() for a in y_test])
        
        

        # # calculating useful metrics
        # fpr = [0,0,0]
        # tpr = [0,0,0]
        # roc_auc = [0,0,0]
        # for i in range(y_pred_list.shape[1]):
        #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_list[:, i])
        #     roc_auc[i] = auc(fpr[i], tpr[i])

        # fpr_grid = np.linspace(0.0, 1.0, 1000)
        # mean_tpr = np.zeros_like(fpr_grid)
        # for i in range(3):
        #     mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
        # mean_tpr /= 3
        
        # fpr, tpr, thresholds = roc_curve(y_test, y_pred_list, pos_label=1)
        results["test"]["fpr"].append(fpr_grid.tolist())
        results["test"]["tpr"].append(mean_tpr.tolist())
        # results["test"]["thresholds"].append(thresholds.tolist())
        results["test"]["auc"].append(auc(fpr_grid, mean_tpr))
        results["test"]["macro_f1"].append(macro_f1)
        auc_all.append(auc(fpr_grid, mean_tpr))

    with open("bert_muli_rf.json", "w") as fp:
        json.dump(results, fp)
    