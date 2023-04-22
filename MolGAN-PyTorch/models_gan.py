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
        
        self.multi_dense_layer = MultiDenseLayer(z_dim, hid_dims, self.activation_f, dropout_rate=dropout_rate)
        self.mha = nn.MultiheadAttention(mha_dim, n_heads, batch_first=True)
        self.multi_dense_layer_2 = MultiDenseLayer(mha_dim, hid_dims_2, self.activation_f, dropout_rate=dropout_rate)

        self.adjM_layer = MultiDenseLayer(hid_dims_2[-1], [256, N*N], self.activation_f)
        self.node_layer = MultiDenseLayer(hid_dims_2[-1], [64, N], self.activation_f)

    def forward(self, z, bert_out):
        out = self.multi_dense_layer(z)
        out = self.mha(out.view(out.shape[0], 1, -1), bert_out, bert_out)[0].view(z.shape[0], -1)
        out = self.multi_dense_layer_2(out)
        adjM_logits = self.adjM_layer(out).view(-1, self.N, self.N)
        adjM_logits = (adjM_logits + adjM_logits.permute(0, 2, 1)) / 2
        node_logits = self.node_layer(out)

        return adjM_logits, node_logits

class Discriminator(nn.Module):
    # Abishek's code
    """Discriminator network with PatchGAN NLP conditioned graph gen."""

    def __init__(self, N, disc_dims, mha_dim, n_heads, dropout_rate=0.):
        super(Discriminator, self).__init__()
        self.activation_f = torch.nn.ReLU()
        hid_dims, hid_dims_2, hid_dims_3 = disc_dims
        self.multi_dense_layer = MultiDenseLayer(N, hid_dims, self.activation_f)
        self.node_dense_layer = MultiDenseLayer(N, [64, hid_dims[-1]], self.activation_f, dropout_rate=dropout_rate)
        self.multi_dense_layer_2 = MultiDenseLayer((N+1)*hid_dims[-1], hid_dims_2, self.activation_f, dropout_rate=dropout_rate)
        self.mha = nn.MultiheadAttention(mha_dim, n_heads, batch_first=True)
        self.multi_dense_layer_3 = MultiDenseLayer(mha_dim, hid_dims_3, self.activation_f, dropout_rate=dropout_rate)

        self.output_layer = nn.Linear(hid_dims_3[-1], 1)

    def forward(self, adj, node, bert_out, activation=None):
        # adj = adj[:, :, :, 1:].permute(0, 3, 1, 2)
        inp = adj
        out = self.multi_dense_layer(inp)
        out_node = self.node_dense_layer(node)
        out = torch.cat([out, out_node.view(out_node.shape[0], 1, -1)], dim=1)
        out = out.view(out.shape[0], -1)
        out = self.multi_dense_layer_2(out)
        out = self.mha(out.view(out.shape[0], 1, -1), bert_out, bert_out)[0].view(out.shape[0], -1)
        out = self.multi_dense_layer_3(out)

        output = self.output_layer(out)
        output = activation(output) if activation is not None else output

        return output, out

def gumbel_sigmoid(logits, t=0.1, eps=1e-20, hard=False):            
    #sample from Gumbel(0, 1)
    uniform1 = torch.rand(logits.shape).to(logits.device)
    uniform2 = torch.rand(logits.shape).to(logits.device)
    
    noise = -torch.log(torch.log(uniform2 + eps)/torch.log(uniform1 + eps) + eps)
    
    #draw a sample from the Gumbel-Sigmoid distribution
    y = torch.sigmoid((logits + noise) / t)
    if len(y.shape) == 3:
        y = (y + y.permute(0, 2, 1)) / 2
    
    if hard:
        #take the sign of the logits
        y_hard = torch.zeros_like(y).to(logits.device)
        y_hard[y >= 0.5] = 1
        y = (y_hard - y).detach() + y

    return y