from torch_geometric.utils import from_networkx
import json
import networkx as nx
from pathlib import Path
from .random_walk import *


def read_data(graphs, union, model_params):
    my_corpus = Path(
        f"""inputs/corpus_nw{model_params['num_walks']}_wl{model_params['walk_length']}_p{model_params['p']}_q{model_params['q']}.txt"""
    )
    if my_corpus.exists():
        print("Reading Corpus from Disk")
        with open(my_corpus._str, "r") as f:
            data = json.load(f)
    else:
        print("generating corpus!")
        data = create_corpus(model_params, graphs)
        with open(my_corpus._str, "w") as f:
            f.write(json.dumps(data))

    netwroks = []
    for graph in graphs:
        g = nx.read_weighted_edgelist(graph, delimiter=" ").to_undirected()
        if not nx.is_weighted(g):
            g.add_weighted_edges_from([(a, b, 1.0) for (a, b) in g.edges])
        pyg_graph = from_networkx(g)
        pyg_graph.edge_weight = pyg_graph.weight
        del pyg_graph.weight
        pyg_graph["node_index"] = list(g.nodes)
        pyg_graph["node_sequence"] = []
        pyg_graph["node_sequence"].append(" ".join(node for node in pyg_graph.node_index))
        netwroks.append(pyg_graph)
    # data = ['YHR147C YJL023C YPL040C YLR091W YPL189C-A YIL157C',
    #         'YBR250W YDL086W YLR159C-A YDR519W YJL052W YBR199W',
    #         'YNL309W YGR161C YDR542W YML109W YJL221C YLR151C']

    # reading union network
    g = nx.read_weighted_edgelist(union, delimiter=" ").to_undirected()
    if not nx.is_weighted(g):
        g.add_weighted_edges_from([(a, b, 1.0) for (a, b) in g.edges])
    g.remove_edges_from(nx.selfloop_edges(g))  # remove existing selfloops first
    # g.add_weighted_edges_from([(n, n, 1.0) for n in g.nodes()])
    pyg_un = from_networkx(g)
    pyg_un.edge_weight = pyg_un.weight
    del pyg_un.weight
    pyg_un["node_index"] = list(g.nodes)
    pyg_un["node_sequence"] = []
    pyg_un["node_sequence"].append(" ".join(node for node in pyg_un.node_index))
    netwroks.append(pyg_un)
    return data, netwroks, pyg_un


# def my_tokenizer(data):
#     tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
#     my_tokens = Path("artifacts/tokenizer.pt")
#     if my_tokens.exists():
#         print("Reading Tokens from Disk")
#         tokenizer = tokenizer.from_file(my_tokens._str)
#     else:
#         print("training tokenizer!")
#         tokenizer.pre_tokenizer = WhitespaceSplit()
#         trainer = trainers.WordLevelTrainer(special_tokens=["[UNK]", "[MASK]", "[CLS]", "[PAD]"])
#         tokenizer.enable_padding()
#         tokenizer.train_from_iterator(map(lambda x: x.split(), data), trainer=trainer)
#         tokenizer.save(my_tokens._str)
#     return tokenizer


# tokenizer.get_vocab()
# output = tokenizer.encode('YHR147C YJL023C YPL040C')
# output.ids


# class MyDataset(Dataset):
#     def __init__(self, src, tokenizer):
#         self.src = [i for i in src]
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.src)

#     def __getitem__(self, idx):
#         src = self.src[idx]
#         return src


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=512):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer("pe", pe)

#     def forward(self, x):
#         x = x + self.pe[: x.size(0), :]
#         return self.dropout(x)
