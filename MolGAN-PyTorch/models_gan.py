import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import GraphConvolution, GraphAggregation, MultiDenseLayer, PositionalEncoder
    
class Generator(nn.Module):
    # Abishek's code
    """Generator network for NLP conditioned graph gen."""

    def __init__(self, N, z_dim, hid_dims, hid_dims_2, mha_dim, n_heads, dropout_rate):
        super(Generator, self).__init__()
        self.N = N
        self.activation_f = torch.nn.Tanh()
        self.multi_dense_layer = MultiDenseLayer(z_dim, hid_dims, self.activation_f)
        
        self.mha = nn.MultiheadAttention(mha_dim, n_heads, batch_first=True)
        
        self.multi_dense_layer_2 = MultiDenseLayer(mha_dim, hid_dims_2, self.activation_f)

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

    def __init__(self, N, conv_dim, m_dim, mha_dim, n_heads, dropout_rate=0.):
        super(Discriminator, self).__init__()
        self.activation_f = torch.nn.Tanh()
        graph_conv_dim, aux_dim, linear_dim = conv_dim
        # discriminator
        self.gcn_layer = GraphConvolution(N, m_dim, graph_conv_dim, dropout_rate)
        self.agg_layer = GraphAggregation(N, graph_conv_dim[-1], aux_dim, self.activation_f, dropout_rate)
        self.mha = nn.MultiheadAttention(mha_dim, n_heads, batch_first=True)
        self.multi_dense_layer = MultiDenseLayer(mha_dim, linear_dim, self.activation_f, dropout_rate=dropout_rate)

        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def forward(self, adj, bert_out, activation=None):
        # adj = adj[:, :, :, 1:].permute(0, 3, 1, 2)
        h_1 = self.gcn_layer(adj)
        h = self.agg_layer(h_1)
        out = self.mha(h.view(h.shape[0], 1, -1), bert_out, bert_out)[0].view(h.shape[0], -1)
        h = torch.cat(out, dim=1)
        h = self.multi_dense_layer(h)

        output = self.output_layer(h)
        output = activation(output) if activation is not None else output

        return output, h
