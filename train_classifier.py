from functools import partial
import torch
import torch.nn as nn
import time
from modeling.models import BinaryClassification
from modeling.data import TrainData, TestData, classifier_collate
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from utils.commons import *
import random

random.seed(1984)
torch.manual_seed(1984)

device = "cuda" if torch.cuda.is_available() else "cpu"


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
        writer.add_scalars("Loss", loss_dict, epoch + 1)


if __name__ == "__main__":

    writer = SummaryWriter(flush_secs=10)

    # Loading pre-trained
    name = "Oct06_19-46-44_s-001"  # pre-trained model name
    checkpoint = torch.load(f"artifacts/{name}_model.pt")
    tokenizer = checkpoint["tokenizer"]
    # Loading pathway data, last token is the label
    with open("/home/bnet/raminasser/node2vec/paths.txt", "r") as f:
        data = json.load(f)
    X = np.array([text.rsplit(" ", 1)[0] for text in data])  # walks only
    y = np.array([int(text.split(" ")[-1]) for text in data])  # labels only
    # prepare for training
    for net in checkpoint["networks"]:
        net["node_tokens"] = torch.tensor(tokenizer.encode(net["node_sequence"][0]).ids).to(device)
        net.edge_index = net.edge_index.to(device)
        net.edge_weight = net.edge_weight.to(device)

    checkpoint["model_params"]["epochs"] = 50
    auc = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()

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
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_list, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
        metrics.RocCurveDisplay.from_predictions(y_test, y_pred_list, name="ROC fold {}".format(i), ax=ax)

    # auc plot
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="",
    )
    ax.legend(loc="lower right")
    plt.show()
    plt.savefig("auc_another_one.pdf")
