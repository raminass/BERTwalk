from mlm import *
import torch
import pandas as pd
import glob

device = "cuda" if torch.cuda.is_available() else "cpu"

# for file in glob.glob("artifacts/tune/*"):
#     name = file.split("/")[2].split(".")[0]
#     checkpoint = torch.load(f"artifacts/tune/{name}.pt")
#     model = TransformerModel(checkpoint["model_params"], checkpoint["networks"]).to(device)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.all_prop_emb()
#     all_emb = []
#     for net in checkpoint["networks"]:
#         x = model.encoder(torch.tensor(net.node_tokens).to(device))
#         all_emb.append(pd.DataFrame(x.detach().cpu().numpy(), index=net.node_index))
#     emb_df = pd.concat(all_emb).drop_duplicates()
#     emb_df.to_csv(f"""outputs/tune/{name}_feat.tsv""", sep="\t")


name = "Oct07_12-11-46_s-001"
checkpoint = torch.load(f"artifacts/{name}_model.pt")
# checkpoint["model_params"]["K"] = 2
# checkpoint["model_params"]["alpha"] = 0.3
model = TransformerModel(checkpoint["model_params"], checkpoint["networks"]).to(device)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()
model.all_prop_emb()

all_emb = []
for net in checkpoint["networks"]:
    x = model.encoder(torch.tensor(net.node_tokens).to(device))
    all_emb.append(pd.DataFrame(x.detach().cpu().numpy(), index=net.node_index))
emb_df = pd.concat(all_emb).drop_duplicates()


# x = model.encoder(torch.tensor(checkpoint["network"].node_tokens).to(device))
# embeddings = model.appnp(x, checkpoint["network"].edge_index.to(device), checkpoint["network"].edge_weight.to(device))
# emb_df = pd.DataFrame(embeddings.detach().cpu().numpy(), index=checkpoint["network"].node_index)

emb_df.to_csv(f"""outputs/{name}_{checkpoint["model_params"]["K"]}_feat.tsv""", sep="\t")


# # post model reading features and prop
# emb_df = pd.read_csv("mlm/outputs/old/gpt_micro_K1_S30.tsv", sep="\t", index_col=0)
# x = emb_df.loc[pyg_graph.node_index, :].values
# x = torch.from_numpy(x).float().to(device)

# embeddings = model.appnp(x, pyg_graph.edge_index.to(device), pyg_graph.edge_weight.to(device))
# emb_df = pd.DataFrame(embeddings.detach().cpu().numpy(), index=pyg_graph.node_index)
# emb_df.to_csv(f"mlm/outputs/gpt_micro_K1_S30.tsv", sep="\t")


# FWD
# output = model.encoder(torch.tensor(list(tokenizer.get_vocab().values())).to(device))
# emb_df = pd.DataFrame(output.detach().cpu().numpy(), index=tokenizer.get_vocab().keys())

# model.eval()
# output, _ = model.fwd_nodes(pyg_graph)
# emb_df = pd.DataFrame(output.detach().cpu().numpy(), index=pyg_graph.node_index)
# emb_df.to_csv(f"outputs/{comment}.tsv", sep="\t")

# checkpoint = torch.load(PATH)
