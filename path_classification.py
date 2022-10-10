# %%
import torch
import torch.nn as nn
import time
from torch_geometric.nn import APPNP
from torch.utils.tensorboard import SummaryWriter
from utils.aux import *
import random
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn import metrics
import matplotlib.pyplot as plt

random.seed(1984)
torch.manual_seed(1984)

device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
class TrainData(Dataset):
    def __init__(self, src, label, tokenizer):
        self.src = [i for i in src]
        self.label = label
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.label[idx]


class TestData(Dataset):
    def __init__(self, src, label, tokenizer):
        self.src = [i for i in src]
        self.label = label
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.label[idx]


def data_collate_fn(dataset_samples_list):
    input = ["[CLS] " + i[0] for i in dataset_samples_list]
    labels = [i[1] for i in dataset_samples_list]
    encoded = tokenizer.encode_batch(input)

    src_mask = torch.zeros(len(encoded[0].ids), len(encoded[0].ids))
    src_mask[1:, 0] = float("-inf")

    input = torch.tensor([i.ids for i in encoded])
    src_key_padding_mask = torch.tensor([i.special_tokens_mask for i in encoded])

    return {
        "input": input,
        "src_mask": src_mask,
        "src_key_padding_mask": src_key_padding_mask.bool(),
        "labels": torch.IntTensor(labels),
    }


# %%
class TransformerModel(nn.Module):
    def __init__(self, model_params, networks):
        super(TransformerModel, self).__init__()
        self.model_type = "Transformer"
        self.ntoken = model_params["ntokens"]
        self.nhead = model_params["nhead"]
        self.nhid = model_params["nhid"]
        self.ninp = model_params["emsize"]
        self.nlayers = model_params["nlayers"]
        self.dropout = model_params["dropout"]
        self.K = model_params["K"]
        self.alpha = model_params["alpha"]
        self.networks = networks

        self.pos_encoder = PositionalEncoding(self.ninp, self.dropout)
        encoder_layers = nn.TransformerEncoderLayer(self.ninp, self.nhead, self.nhid, self.dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.nlayers)
        self.encoder = nn.Embedding(self.ntoken, self.ninp)
        self.decoder = nn.Linear(self.ninp, self.ntoken)

        # propagation layer
        self.appnp = nn.ModuleList(
            [APPNP(K=self.K, alpha=self.alpha, cached=True, add_self_loops=False) for i in self.networks]
        )
        self.init_weights()

    def all_prop_emb(self):
        for i, net in enumerate(self.networks):
            x = self.encoder(torch.tensor(net.node_tokens).to(device)).detach()
            x = self.appnp[i](x, net.edge_index.to(device), net.edge_weight.to(device))
            self.encoder.weight.data[net.node_tokens] = x

    def alt_prop_emb(self, epoch):
        i = epoch % len(self.networks)
        net = self.networks[i]
        x = self.encoder(torch.tensor(net.node_tokens).to(device)).detach()
        x = self.appnp[i](x, net.edge_index.to(device), net.edge_weight.to(device))
        self.encoder.weight.data[net.node_tokens] = x

    def fwd_nodes(self, netwrok):

        src = self.encoder(torch.tensor(netwrok.node_tokens).to(device)) * math.sqrt(self.ninp)
        src = self.transformer_encoder(src)
        emb = self.appnp(src, netwrok.edge_index.to(device), netwrok.edge_weight.to(device))
        emb = torch.div(emb, torch.norm(emb, dim=1)[:, None])
        return emb, None

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask, src_key_padding_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        # output = self.decoder(output)
        return output


class BinaryClassification(nn.Module):
    def __init__(self, checkpoint):
        super(BinaryClassification, self).__init__()
        self.pre_class = TransformerModel(checkpoint["model_params"], checkpoint["networks"])
        self.pre_class.load_state_dict(checkpoint["model_state_dict"])
        self.layer_out = nn.Linear(checkpoint["model_params"]["emsize"], 1)

    def forward(self, src, src_mask, src_key_padding_mask):
        x_cls = self.pre_class(src, src_mask, src_key_padding_mask)
        return self.layer_out(x_cls[:, 0, :])  # take only cls vector


# %%
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


# %%
if __name__ == "__main__":
    global tokenizer
    global pyg_graphs
    writer = SummaryWriter(flush_secs=10)
    name = "Oct06_19-46-44_s-001"
    checkpoint = torch.load(f"artifacts/{name}_model.pt")

    tokenizer = checkpoint["tokenizer"]
    with open("/home/bnet/raminasser/node2vec/longer_paths.txt", "r") as f:
        data = json.load(f)

    X = np.array([text.rsplit(" ", 1)[0] for text in data])  # walks only
    y = np.array([int(text.split(" ")[-1]) for text in data])  # labels only

    checkpoint["model_params"]["epochs"] = 50
    auc = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=38)

    for i, (train, test) in enumerate(skf.split(X, y)):
        class_model = BinaryClassification(checkpoint)
        class_model.to(device)
        train_data = TrainData(X[train].tolist(), y[train].tolist(), tokenizer)
        test_data = TestData(X[test].tolist(), y[test].tolist(), tokenizer)

        train_loader = DataLoader(
            dataset=train_data,
            batch_size=checkpoint["model_params"]["batch_size"],
            shuffle=True,
            collate_fn=data_collate_fn,
        )
        test_loader = DataLoader(dataset=test_data, batch_size=1, collate_fn=data_collate_fn)
        train_fun(class_model, train_loader, checkpoint["model_params"])

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
        # print(metrics.classification_report(y_test, y_pred_list))

        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_list, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)

        metrics.RocCurveDisplay.from_predictions(y_test, y_pred_list, name="ROC fold {}".format(i), ax=ax)

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
    plt.savefig("auc_longer_paths.pdf")
    # print(auc)
