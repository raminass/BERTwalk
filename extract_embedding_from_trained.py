from modeling.models import TransformerModel
import torch
import pandas as pd
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():

    parser = argparse.ArgumentParser(description="Extract the trained BBERTwalk embedding.")
    parser.add_argument("--model_name", type=str, default="Oct06_19-46-44_s-001", help="Trained model name.")
    return parser.parse_args()


args = parse_args()
name = args.model_name
checkpoint = torch.load(f"artifacts/{name}_model.pt")
model = TransformerModel(checkpoint["model_params"], checkpoint["networks"]).to(device)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()
for net in checkpoint["networks"]:
    net["node_tokens"] = torch.tensor(checkpoint["tokenizer"].encode(net["node_sequence"][0]).ids).to(device)
    net.edge_index = net.edge_index.to(device)
    net.edge_weight = net.edge_weight.to(device)

model.all_prop_emb()

all_emb = []
for net in checkpoint["networks"]:
    x = model.encoder(net.node_tokens)
    all_emb.append(pd.DataFrame(x.detach().cpu().numpy(), index=net.node_index))
emb_df = pd.concat(all_emb).drop_duplicates()

emb_df.to_csv(f"""outputs/{name}_{checkpoint["model_params"]["K"]}_feat.tsv""", sep="\t")
