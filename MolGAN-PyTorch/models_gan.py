import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import numpy as np
from layers import GraphConvolution2, GraphAggregation2, MultiGraphConvolutionLayers2, MultiDenseLayer
    
class Generator2(nn.Module):
    # Abishek's code
    """Generator network for NLP conditioned graph gen."""

    def __init__(self, N, z_dim, hid_dims, hid_dims_2, dropout_rate):
        super(Generator2, self).__init__()
        self.N = N
        self.activation_f = torch.nn.Tanh()
        self.multi_dense_layer = MultiDenseLayer(z_dim, hid_dims, self.activation_f)
        # self.bert = BertModel.from_pretrained('bert-base-cased')
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        
        self.mha = nn.MultiheadAttention(768, 8, batch_first=True)
        
        self.multi_dense_layer_2 = MultiDenseLayer(hid_dims[-1], hid_dims_2, self.activation_f)

        self.adjM_layer = nn.Linear(hid_dims_2[-1], N*N)
        # self.node_layer = nn.Linear(hid_dims_2[-1], N)
        self.dropoout = nn.Dropout(p=dropout_rate)

    def forward(self, z, bert_out):
        out = self.multi_dense_layer(z)
        # bert_out = self.bert(input_ids, attention_mask=mask).last_hidden_state[:,:self.N,:]
        out = self.mha(out, bert_out, bert_out)[0].view(z.shape[0], -1)
        out = self.multi_dense_layer_2(out)
        adjM_logits = self.adjM_layer(out).view(-1, self.N, self.N)
        adjM_logits = (adjM_logits + adjM_logits.permute(0, 2, 1)) / 2
        adjM_logits = self.dropoout(adjM_logits)

        return adjM_logits

class Discriminator2(nn.Module):
    # Abishek's code
    """Discriminator network with PatchGAN NLP conditioned graph gen."""

    def __init__(self, conv_dim, m_dim, b_dim, dropout_rate=0.):
        super(Discriminator2, self).__init__()
        self.activation_f = torch.nn.Tanh()
        graph_conv_dim, aux_dim, linear_dim = conv_dim
        # discriminator
        self.gcn_layer = GraphConvolution2(m_dim, graph_conv_dim, dropout_rate)
        self.agg_layer = GraphAggregation2(graph_conv_dim[-1], aux_dim, self.activation_f, dropout_rate)
        self.multi_dense_layer = MultiDenseLayer(aux_dim+b_dim, linear_dim, self.activation_f, dropout_rate=dropout_rate)

        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def forward(self, adj, node, bert_out, activation=None):
        adj = adj[:, :, :, 1:].permute(0, 3, 1, 2)
        h_1 = self.gcn_layer(node, adj)
        h = self.agg_layer(node, h_1)
        h = torch.cat([h, bert_out], dim=1)
        h = self.multi_dense_layer(h)

        output = self.output_layer(h)
        output = activation(output) if activation is not None else output

        return output, h



class Generator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, dropout_rate):
        super(Generator, self).__init__()
        self.activation_f = torch.nn.Tanh()
        self.multi_dense_layer = MultiDenseLayer(z_dim, conv_dims, self.activation_f)

        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes

        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * nodes)
        self.dropoout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.multi_dense_layer(x)
        edges_logits = self.edges_layer(output).view(-1, self.edges, self.vertexes, self.vertexes)
        edges_logits = (edges_logits + edges_logits.permute(0, 1, 3, 2)) / 2
        edges_logits = self.dropoout(edges_logits.permute(0, 2, 3, 1))

        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropoout(nodes_logits.view(-1, self.vertexes, self.nodes))

        return edges_logits, nodes_logits

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, conv_dim, m_dim, b_dim, with_features=False, f_dim=0, dropout_rate=0.):
        super(Discriminator, self).__init__()
        self.activation_f = torch.nn.Tanh()
        graph_conv_dim, aux_dim, linear_dim = conv_dim
        # discriminator
        self.gcn_layer = GraphConvolution(m_dim, graph_conv_dim, b_dim, with_features, f_dim, dropout_rate)
        self.agg_layer = GraphAggregation(graph_conv_dim[-1] + m_dim, aux_dim, self.activation_f, with_features, f_dim,
                                          dropout_rate)
        self.multi_dense_layer = MultiDenseLayer(aux_dim, linear_dim, self.activation_f, dropout_rate=dropout_rate)

        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def forward(self, adj, hidden, node, activation=None):
        adj = adj[:, :, :, 1:].permute(0, 3, 1, 2)
        h_1 = self.gcn_layer(node, adj, hidden)
        h = self.agg_layer(h_1, node, hidden)
        h = self.multi_dense_layer(h)

        output = self.output_layer(h)
        output = activation(output) if activation is not None else output

        return output, h
