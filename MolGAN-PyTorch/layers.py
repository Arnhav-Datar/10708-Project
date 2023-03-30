import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.nn.modules.module import Module

class PositionalEncoder(nn.Module):
    # Abishek's code
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], \
        requires_grad=False).cuda()
        return x

class GraphConvolutionLayer2(Module):
    # Abishek's code
    def __init__(self, in_features, u, activation, dropout_rate=0.):
        super(GraphConvolutionLayer2, self).__init__()
        self.u = u
        self.adj_list = nn.Linear(in_features, u)
        self.linear_2 = nn.Linear(in_features, u)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, n_tensor, adj_tensor, h_tensor=None):
        if h_tensor is not None:
            annotations = n_tensor + h_tensor
        else:
            annotations = n_tensor

        output = self.adj_list(annotations)
        output = torch.matmul(adj_tensor, output)
        out_linear_2 = self.linear_2(annotations)
        output = output + out_linear_2
        output = self.activation(output) if self.activation is not None else output
        output = self.dropout(output)
        return output

    
class MultiGraphConvolutionLayers2(Module):
    # Abishek's code
    def __init__(self, in_features, conv_hid_dims, activation, dropout_rate=0.):
        super(MultiGraphConvolutionLayers2, self).__init__()
        self.conv_nets = nn.ModuleList()
        self.units = conv_hid_dims
    
        for u0, u1 in zip([in_features] + self.units[:-1], self.units):
            self.conv_nets.append(GraphConvolutionLayer2(u0, u1, activation, dropout_rate))

    def forward(self, n_tensor, adj_tensor, h_tensor=None):
        hidden_tensor = h_tensor
        for conv_idx in range(len(self.units)):
            hidden_tensor = self.conv_nets[conv_idx](n_tensor, adj_tensor, hidden_tensor)
        return hidden_tensor


class GraphConvolution2(Module):
    # Abishek's code
    def __init__(self, in_features, graph_conv_units, dropout_rate=0.):
        super(GraphConvolution2, self).__init__()
        self.in_features = in_features
        self.graph_conv_units = graph_conv_units
        self.activation_f = torch.nn.Tanh()
        self.multi_graph_convolution_layers = \
            MultiGraphConvolutionLayers(in_features, self.graph_conv_units, self.activation_f, dropout_rate)

    def forward(self, n_tensor, adj_tensor, h_tensor=None):
        output = self.multi_graph_convolution_layers(n_tensor, adj_tensor, h_tensor)
        return output

class GraphAggregation2(Module):
    # Abishek's code
    def __init__(self, in_features, aux_units, activation, dropout_rate=0.):
        super(GraphAggregation2, self).__init__()
        self.activation = activation
        self.i = nn.Sequential(nn.Linear(in_features, aux_units),
                                nn.Sigmoid())
        j_layers = [nn.Linear(in_features, aux_units)]
        if self.activation is not None:
            j_layers.append(self.activation)
        self.j = nn.Sequential(*j_layers)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, n_tensor, out_tensor, h_tensor=None):
        out_tensor = out_tensor + n_tensor
        if h_tensor is not None:
            annotations = torch.cat((out_tensor, h_tensor), -1)
        # The i here seems to be an attention.
        i = self.i(annotations)
        j = self.j(annotations)
        output = torch.sum(torch.mul(i, j), 1)
        if self.activation is not None:
            output = self.activation(output)
        output = self.dropout(output)

        return output





class GraphConvolutionLayer(Module):
    def __init__(self, in_features, u, activation, edge_type_num, dropout_rate=0.):
        super(GraphConvolutionLayer, self).__init__()
        self.edge_type_num = edge_type_num
        self.u = u
        self.adj_list = nn.ModuleList()
        for _ in range(self.edge_type_num):
            self.adj_list.append(nn.Linear(in_features, u))
        self.linear_2 = nn.Linear(in_features, u)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, n_tensor, adj_tensor, h_tensor=None):
        if h_tensor is not None:
            annotations = torch.cat((n_tensor, h_tensor), -1)
        else:
            annotations = n_tensor

        output = torch.stack([self.adj_list[i](annotations) for i in range(self.edge_type_num)], 1)
        output = torch.matmul(adj_tensor, output)
        out_sum = torch.sum(output, 1)
        out_linear_2 = self.linear_2(annotations)
        output = out_sum + out_linear_2
        output = self.activation(output) if self.activation is not None else output
        output = self.dropout(output)
        return output

class MultiGraphConvolutionLayers(Module):
    def __init__(self, in_features, units, activation, edge_type_num, with_features=False, f=0, dropout_rate=0.):
        super(MultiGraphConvolutionLayers, self).__init__()
        self.conv_nets = nn.ModuleList()
        self.units = units
        in_units = []
        if with_features:
            for i in range(len(self.units)):
                in_units = list([x + in_features for x in self.units])
            for u0, u1 in zip([in_features+f] + in_units[:-1], self.units):
                self.conv_nets.append(GraphConvolutionLayer(u0, u1, activation, edge_type_num, dropout_rate))
        else:
            for i in range(len(self.units)):
                in_units = list([x + in_features for x in self.units])
            for u0, u1 in zip([in_features] + in_units[:-1], self.units):
                self.conv_nets.append(GraphConvolutionLayer(u0, u1, activation, edge_type_num, dropout_rate))

    def forward(self, n_tensor, adj_tensor, h_tensor=None):
        hidden_tensor = h_tensor
        for conv_idx in range(len(self.units)):
            hidden_tensor = self.conv_nets[conv_idx](n_tensor, adj_tensor, hidden_tensor)
        return hidden_tensor

class GraphConvolution(Module):
    def __init__(self, in_features, graph_conv_units, edge_type_num, with_features=False, f_dim=0, dropout_rate=0.):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.graph_conv_units = graph_conv_units
        self.activation_f = torch.nn.Tanh()
        self.multi_graph_convolution_layers = \
            MultiGraphConvolutionLayers(in_features, self.graph_conv_units, self.activation_f, edge_type_num,
                                        with_features, f_dim, dropout_rate)

    def forward(self, n_tensor, adj_tensor, h_tensor=None):
        output = self.multi_graph_convolution_layers(n_tensor, adj_tensor, h_tensor)
        return output


class GraphConvolution3(Module):
    def __init__(self, in_features, out_feature_list, b_dim, dropout):
        super(GraphConvolution3, self).__init__()
        self.in_features = in_features
        self.out_feature_list = out_feature_list

        self.linear1 = nn.Linear(in_features, out_feature_list[0])
        self.linear2 = nn.Linear(out_feature_list[0], out_feature_list[1])

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, adj, activation=None):
        # input : 16x9x9
        # adj : 16x4x9x9

        hidden = torch.stack([self.linear1(inputs) for _ in range(adj.size(1))], 1)
        hidden = torch.einsum('bijk,bikl->bijl', (adj, hidden))
        hidden = torch.sum(hidden, 1) + self.linear1(inputs)
        hidden = activation(hidden) if activation is not None else hidden
        hidden = self.dropout(hidden)

        output = torch.stack([self.linear2(hidden) for _ in range(adj.size(1))], 1)
        output = torch.einsum('bijk,bikl->bijl', (adj, output))
        output = torch.sum(output, 1) + self.linear2(hidden)
        output = activation(output) if activation is not None else output
        output = self.dropout(output)

        return output


class GraphAggregation(Module):
    def __init__(self, in_features, aux_units, activation, with_features=False, f_dim=0,
                 dropout_rate=0.):
        super(GraphAggregation, self).__init__()
        self.with_features = with_features
        self.activation = activation
        if self.with_features:
            self.i = nn.Sequential(nn.Linear(in_features+f_dim, aux_units),
                                   nn.Sigmoid())
            j_layers = [nn.Linear(in_features+f_dim, aux_units)]
            if self.activation is not None:
                j_layers.append(self.activation)
            self.j = nn.Sequential(*j_layers)
        else:
            self.i = nn.Sequential(nn.Linear(in_features, aux_units),
                                   nn.Sigmoid())
            j_layers = [nn.Linear(in_features, aux_units)]
            if self.activation is not None:
                j_layers.append(self.activation)
            self.j = nn.Sequential(*j_layers)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, n_tensor, out_tensor, h_tensor=None):
        if h_tensor is not None:
            annotations = torch.cat((out_tensor, h_tensor, n_tensor), -1)
        else:
            annotations = torch.cat((out_tensor, n_tensor), -1)
        # The i here seems to be an attention.
        i = self.i(annotations)
        j = self.j(annotations)
        output = torch.sum(torch.mul(i, j), 1)
        if self.activation is not None:
            output = self.activation(output)
        output = self.dropout(output)

        return output


class GraphAggregation3(Module):

    def __init__(self, in_features, out_features, b_dim, dropout):
        super(GraphAggregation3, self).__init__()
        self.sigmoid_linear = nn.Sequential(nn.Linear(in_features+b_dim, out_features),
                                            nn.Sigmoid())
        self.tanh_linear = nn.Sequential(nn.Linear(in_features+b_dim, out_features),
                                         nn.Tanh())
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, activation):
        i = self.sigmoid_linear(inputs)
        j = self.tanh_linear(inputs)
        output = torch.sum(torch.mul(i, j), 1)
        output = activation(output) if activation is not None else output
        output = self.dropout(output)

        return output


class MultiDenseLayer(Module):
    def __init__(self, aux_unit, linear_units, activation=None, dropout_rate=0.):
        super(MultiDenseLayer, self).__init__()
        layers = []
        for c0, c1 in zip([aux_unit] + linear_units[:-1], linear_units):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Dropout(dropout_rate))
            if activation is not None:
                layers.append(activation)
        self.linear_layer = nn.Sequential(*layers)

    def forward(self, inputs):
        h = self.linear_layer(inputs)
        return h
