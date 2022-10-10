# %%
import torch
import torch.nn as nn
import time
from torch_geometric.nn import APPNP
from torch.utils.tensorboard import SummaryWriter
from utils.aux import *
import random

random.seed(1984)
torch.manual_seed(1984)

device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
def data_collate_fn(dataset_samples_list):
    input = ["[CLS] " + i for i in dataset_samples_list]
    encoded = tokenizer.encode_batch(input)

    src_mask = torch.zeros(len(encoded[0].ids), len(encoded[0].ids))
    src_mask[1:, 0] = float("-inf")

    input = torch.tensor([i.ids for i in encoded])
    src_key_padding_mask = torch.tensor([i.special_tokens_mask for i in encoded])

    return {"input": input, "src_mask": src_mask, "src_key_padding_mask": src_key_padding_mask.bool()}


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
        output = self.decoder(output)
        return output


# %%
def train(model, dataloader, model_params):
    lr = model_params["learning_rate"]
    mask_token_id = tokenizer.get_vocab()["[MASK]"]
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    best_model = None
    for epoch in range(model_params["epochs"]):
        total_loss = 0
        epoch_start_time = time.time()
        model.all_prop_emb()
        for b, batch in enumerate(dataloader):
            optim.zero_grad()
            input = batch["input"].clone()
            labels = batch["input"].clone()
            src_mask = batch["src_mask"]
            src_key_padding_mask = batch["src_key_padding_mask"]

            rand_mask = ~batch["input"].bool()
            for i, row in enumerate(
                torch.randint(
                    1,
                    batch["input"].shape[1],
                    (batch["input"].shape[0], int(batch["input"].shape[1] * model_params["mask_rate"])),
                )
            ):
                rand_mask[i, row] = True

            mask_idx = (rand_mask.flatten() == True).nonzero().view(-1)
            input = input.flatten()
            input[mask_idx] = mask_token_id
            input = input.view(batch["input"].size())
            labels[input != mask_token_id] = -100

            out = model(input.to(device), src_mask.to(device), src_key_padding_mask.to(device))
            loss = criterion(out.view(-1, model_params["ntokens"]), labels.view(-1).to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optim.step()

            total_loss += loss.item()

        elapsed = time.time() - epoch_start_time
        loss_dict = {}
        val_loss = total_loss / (len(dataloader))
        loss_dict["total"] = val_loss

        print("-" * 89)
        print(f"| epoch {1+epoch:3d} | time: {elapsed:5.2f}s | " f"loss {val_loss:5.2f} | lr {lr:02.5f}")
        print("-" * 89)
        writer.add_scalars("Loss", loss_dict, epoch + 1)

        if total_loss < best_loss:
            best_loss = total_loss
            best_model = model.state_dict()

    return best_model


# %%
if __name__ == "__main__":
    global tokenizer
    global pyg_graphs
    writer = SummaryWriter(flush_secs=10)

    model_params = {
        "batch_size": 64,
        "emsize": 128,
        "nhid": 200,
        "nlayers": 4,
        "nhead": 4,
        "dropout": 0.0,
        "learning_rate": 0.0005,
        "epochs": 500,
        "K": 2,
        "alpha": 0.3,
        "mask_rate": 0.4,
        "q": 1,
        "p": 1,
        "walk_length": 10,
        "num_walks": 10,
        "weighted": 1,
        "directed": 0,
    }

    graphs = ["inputs/Costanzo-2016.txt", "inputs/Hu-2007.txt", "inputs/Krogan-2006.txt"]
    union = "inputs/union.txt"  # calculated propabilistically
    # data, pyg_graphs, _ = read_data(f"inputs/corpus_{model_params['corp_size']}.txt", graphs, union, model_params)
    data, pyg_graphs, _ = read_data(graphs, union, model_params)
    tokenizer = my_tokenizer(data)
    for net in pyg_graphs:
        net["node_tokens"] = tokenizer.encode(net["node_sequence"][0]).ids

    model_params["ntokens"] = tokenizer.get_vocab_size()

    writer.add_hparams(model_params, {"hparam/loss": 1})

    model = TransformerModel(model_params, pyg_graphs).to(device)
    dataset = MyDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=model_params["batch_size"], collate_fn=data_collate_fn, shuffle=True)
    # Training Model
    best_model = train(model, dataloader, model_params)
    print("Finish Training")

    print("saving best model!")
    torch.save(
        {
            "model_params": model_params,
            "model_state_dict": best_model,
            "networks": pyg_graphs,
            "tokenizer": tokenizer,
        },
        f"artifacts/{writer.log_dir.split('/')[1]}_model.pt",
    )
