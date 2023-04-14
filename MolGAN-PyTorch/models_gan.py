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
        hid_dims, hid_dims_2 = disc_dims
        self.multi_dense_layer = MultiDenseLayer(N*N, hid_dims, self.activation_f)
        self.mha = nn.MultiheadAttention(mha_dim, n_heads, batch_first=True)
        self.multi_dense_layer_2 = MultiDenseLayer(mha_dim, hid_dims_2, self.activation_f, dropout_rate=dropout_rate)

        self.output_layer = nn.Linear(hid_dims_2[-1], 1)

    def forward(self, adj, bert_out, activation=None):
        # adj = adj[:, :, :, 1:].permute(0, 3, 1, 2)
        inp = adj.view(adj.shape[0], -1)
        out = self.multi_dense_layer(inp)
        out = self.mha(out.view(out.shape[0], 1, -1), bert_out, bert_out)[0].view(out.shape[0], -1)
        out = self.multi_dense_layer_2(out)

        output = self.output_layer(out)
        output = activation(output) if activation is not None else output

        return output, out
