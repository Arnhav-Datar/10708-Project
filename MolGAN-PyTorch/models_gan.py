import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import GraphConvolution, GraphAggregation, MultiDenseLayer, PositionalEncoder
    
class Generator(nn.Module):
    # Abishek's code
    """Generator network for NLP conditioned graph gen."""

    def __init__(self, N, z_dim, gen_dims, mha_dim, n_heads, dropout_rate):
        super(Generator, self).__init__()
        self.N = N
        self.activation_f = torch.nn.ReLU()
        hid_dims, hid_dims_2 = gen_dims
        
        self.multi_dense_layer = MultiDenseLayer(z_dim, hid_dims, self.activation_f)
        self.mha = nn.MultiheadAttention(mha_dim, n_heads, batch_first=True)
        self.multi_dense_layer_2 = MultiDenseLayer(mha_dim, hid_dims_2, self.activation_f, dropout_rate=dropout_rate)

        self.adjM_layer = nn.Linear(hid_dims_2[-1], N*N)
        self.dropoout = nn.Dropout(p=dropout_rate)

    def forward(self, z, bert_out):
        out = self.multi_dense_layer(z)
        out = self.mha(out.view(out.shape[0], 1, -1), bert_out, bert_out)[0].view(z.shape[0], -1)
        out = self.multi_dense_layer_2(out)
        adjM_logits = self.adjM_layer(out).view(-1, self.N, self.N)
        adjM_logits = (adjM_logits + adjM_logits.permute(0, 2, 1)) / 2
        adjM_logits = self.dropoout(adjM_logits)

        return adjM_logits

class Discriminator(nn.Module):
    # Abishek's code
    """Discriminator network with PatchGAN NLP conditioned graph gen."""

    def __init__(self, N, disc_dims, mha_dim, n_heads, dropout_rate=0.):
        super(Discriminator, self).__init__()
        self.activation_f = torch.nn.ReLU()
        hid_dims, hid_dims_2, hid_dims_3 = disc_dims
        self.multi_dense_layer = MultiDenseLayer(N, hid_dims, self.activation_f)
        self.multi_dense_layer_2 = MultiDenseLayer(N*hid_dims[-1], hid_dims_2, self.activation_f, dropout_rate=dropout_rate)
        self.mha = nn.MultiheadAttention(mha_dim, n_heads, batch_first=True)
        self.multi_dense_layer_3 = MultiDenseLayer(mha_dim, hid_dims_3, self.activation_f, dropout_rate=dropout_rate)

        self.output_layer = nn.Linear(hid_dims_3[-1], 1)

    def forward(self, adj, bert_out, activation=None):
        # adj = adj[:, :, :, 1:].permute(0, 3, 1, 2)
        inp = adj
        out = self.multi_dense_layer(inp)
        out = out.view(out.shape[0], -1)
        out = self.multi_dense_layer_2(out)
        out = self.mha(out.view(out.shape[0], 1, -1), bert_out, bert_out)[0].view(out.shape[0], -1)
        out = self.multi_dense_layer_3(out)

        output = self.output_layer(out)
        output = activation(output) if activation is not None else output

        return output, out

class RewardNet(nn.Module):
    def __init__(self, N):
        super(RewardNet, self).__init__()
        self.N = N
        self.node_cnt = nn.Sequential(
            nn.Linear(N*N, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # self.edge_cnt = nn.Sequential(
        #     nn.Linear(dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1),
        #     nn.Sigmoid()
        # )

    def forward(self, out):
        # `out` is generator output
        # `out` has shape [batch_size, N, N]

        node_cnt = self.node_cnt(out.view(out.shape[0], -1))
        # `node_cnt` has shape [batch_size, 1] and is normalized to [0,1]

        # edge_cnt = self.edge_cnt(out)
        # `edge_cnt` has shape [batch_size, 1]

        return node_cnt.view(-1)