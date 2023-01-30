import torch
import torch.nn as nn
from torch_geometric.nn import APPNP
import math
from .layers import PositionalEncoding


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
            x = self.encoder(net.node_tokens).detach()
            x = self.appnp[i](x, net.edge_index, net.edge_weight)
            self.encoder.weight.data[net.node_tokens] = x

    def alt_prop_emb(self, epoch):
        i = epoch % len(self.networks)
        net = self.networks[i]
        x = self.encoder(net.node_tokens).detach()
        x = self.appnp[i](x, net.edge_index, net.edge_weight)
        self.encoder.weight.data[net.node_tokens] = x

    def fwd_nodes(self, netwrok):

        src = self.encoder(torch.tensor(netwrok.node_tokens)) * math.sqrt(self.ninp)
        src = self.transformer_encoder(src)
        emb = self.appnp(src, netwrok.edge_index, netwrok.edge_weight)
        emb = torch.div(emb, torch.norm(emb, dim=1)[:, None])
        return emb, None

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def hidden_cls(self, src, src_mask, src_key_padding_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        return output[:, 0, :]  # take only cls vector

    def forward(self, src, src_mask, src_key_padding_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        output = self.decoder(output)
        return output


class BinaryClassification(nn.Module):
    def __init__(self, checkpoint):
        super(BinaryClassification, self).__init__()
        self.pre_class = TransformerModel(checkpoint["model_params"], checkpoint["networks"])
        self.pre_class.load_state_dict(checkpoint["model_state_dict"])  # loading pre-trained part
        self.layer_out = nn.Linear(checkpoint["model_params"]["emsize"], 1)

    def forward(self, src, src_mask, src_key_padding_mask):
        x_cls = self.pre_class.hidden_cls(src, src_mask, src_key_padding_mask)
        return self.layer_out(x_cls)


class TrainedBert(nn.Module):
    def __init__(self, checkpoint):
        super(TrainedBert, self).__init__()
        self.pre_class = TransformerModel(checkpoint["model_params"], checkpoint["networks"])
        self.pre_class.load_state_dict(checkpoint["model_state_dict"])  # loading pre-trained part
        # self.layer_out = nn.Linear(checkpoint["model_params"]["emsize"], 3)

    def forward(self, src, src_mask, src_key_padding_mask):
        x_cls = self.pre_class.hidden_cls(src, src_mask, src_key_padding_mask)
        # return self.layer_out(x_cls)
        return x_cls


class MultiClassification(nn.Module):
    def __init__(self, checkpoint):
        super(MultiClassification, self).__init__()
        # self.pre_class = TransformerModel(checkpoint["model_params"], checkpoint["networks"])
        # self.pre_class.load_state_dict(checkpoint["model_state_dict"])  # loading pre-trained part
        self.layer_out = nn.Linear(checkpoint["model_params"]["emsize"], 3)

    def forward(self, x_cls):
        return self.layer_out(x_cls)