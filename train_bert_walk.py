from functools import partial
import torch
import torch.nn as nn
import time
from modeling.models import TransformerModel
from modeling.data import bert_walk_collate, BertWalkDataset
from modeling.tokenizer import bert_walk_tokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.commons import *
import random

random.seed(1984)
torch.manual_seed(1984)

device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
def parse_args():

    parser = argparse.ArgumentParser(description="Train the BBERTwalk model on MLM task.")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of batch.")
    parser.add_argument("--emsize", type=int, default=128, help="Dim of embbeding.")
    parser.add_argument("--nhid", type=int, default=200, help="Num of hidden dim.")
    parser.add_argument("--nlayers", type=int, default=4, help="Num of transformer layers.")
    parser.add_argument("--nhead", type=int, default=4, help="Num of attention heads in transformer.")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout.")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=500, help="Num of training epochs.")
    parser.add_argument("--K", type=int, default=1, help="Num of propagation iterations.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Reset factor RWR.")
    parser.add_argument("--mask_rate", type=float, default=0.2, help="masking rate MLM.")
    parser.add_argument("--p", type=float, default=1, help="Return hyperparameter. Default is 1.")
    parser.add_argument("--q", type=float, default=1, help="Inout hyperparameter. Default is 1.")
    parser.add_argument("--walk_length", type=int, default=10, help="Length of random walk.")
    parser.add_argument("--num_walks", type=int, default=10, help="Num of walks from each node.")

    return parser.parse_args()


def train_bert_walk(model, dataloader, model_params, tokenizer):
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

    writer = SummaryWriter(flush_secs=10)
    args = parse_args()
    model_params = {
        "batch_size": args.batch_size,
        "emsize": args.emsize,
        "nhid": args.nhid,
        "nlayers": args.nlayers,
        "nhead": args.nhead,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "K": args.K,
        "alpha": args.alpha,
        "mask_rate": args.mask_rate,
        "q": args.q,
        "p": args.p,
        "walk_length": args.walk_length,
        "num_walks": args.num_walks,
        "weighted": 1,
        "directed": 0,
    }

    # input networks
    graphs = ["inputs/Costanzo-2016.txt", "inputs/Hu-2007.txt", "inputs/Krogan-2006.txt"]
    union = "inputs/union.txt"  # calculated propabilistically

    # reading networks and corpus
    data, pyg_graphs, _ = read_data(graphs, union, model_params)
    tokenizer = bert_walk_tokenizer(data)  # loading Tokenizer

    # tokenizing each network nodes and copy to device
    for net in pyg_graphs:
        net["node_tokens"] = torch.tensor(tokenizer.encode(net["node_sequence"][0]).ids).to(device)
        net.edge_index = net.edge_index.to(device)
        net.edge_weight = net.edge_weight.to(device)
    model_params["ntokens"] = tokenizer.get_vocab_size()
    writer.add_hparams(model_params, {"hparam/loss": 1})  # Hparam logging to TB

    # Building model
    model = TransformerModel(model_params, pyg_graphs).to(device)
    dataset = BertWalkDataset(data, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=model_params["batch_size"],
        collate_fn=partial(bert_walk_collate, tokenizer=tokenizer),
        shuffle=True,
    )

    # Training Model
    best_model = train_bert_walk(model, dataloader, model_params, tokenizer)
    print("Finish Training")

    # saving model artifacts
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
