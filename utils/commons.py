from torch_geometric.utils import from_networkx
import json
import networkx as nx
from pathlib import Path
from .random_walk import *


def read_data(graphs, model_params):
    my_corpus = Path(
        f"""inputs/corpus_nw{model_params['num_walks']}_wl{model_params['walk_length']}_{model_params['organism']}.txt"""
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

    # # reading union network
    # g = nx.read_weighted_edgelist(union, delimiter=" ").to_undirected()
    # if not nx.is_weighted(g):
    #     g.add_weighted_edges_from([(a, b, 1.0) for (a, b) in g.edges])
    # g.remove_edges_from(nx.selfloop_edges(g))  # remove existing selfloops first
    # # g.add_weighted_edges_from([(n, n, 1.0) for n in g.nodes()])
    # pyg_un = from_networkx(g)
    # pyg_un.edge_weight = pyg_un.weight
    # del pyg_un.weight
    # pyg_un["node_index"] = list(g.nodes)
    # pyg_un["node_sequence"] = []
    # pyg_un["node_sequence"].append(" ".join(node for node in pyg_un.node_index))
    # netwroks.append(pyg_un)
    return data, netwroks
