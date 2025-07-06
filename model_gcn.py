import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv, FAConv
from torch.nn import Parameter
import numpy as np, itertools, random, copy, math
import math
import scipy.sparse as sp
# import ipdb
# from HypergraphConv import HypergraphConv
from torch_geometric.nn import GCNConv
from itertools import permutations
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import add_self_loops
from graphgcn import GraphGCN



class GraphConvolution(nn.Module):
    """
    Implements common graph convolution operation with special features such as variant mode and residual connection.
    """
    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        # If using variant mode, double the input feature dimension
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        # Weight parameter: input feature size x output feature size
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        # Initialize weight parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights uniformly
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        """
            Parameters:
                - input: Input feature matrix (N x in_features)
                - adj: Adjacency matrix of the graph (N x N), can be sparse
                - h0: Initial node features (N x in_features)
                - lamda: Regularization parameter
                - alpha: Balancing coefficient to mix neighbor information and original features
                - l: Another parameter used to compute `theta`

            Returns:
                - output: Output feature matrix after graph convolution (N x out_features)
        """
        # Compute theta to balance neighbor info and original info
        theta = math.log(lamda / l + 1)
        # Compute weighted features from neighbor nodes (core of convolution)
        hi = torch.spmm(adj, input)  # Sparse matrix multiplication with adjacency matrix
        # Variant mode: concatenate current node features and initial features
        if self.variant:
            support = torch.cat([hi, h0], 1)  # Concatenate neighbor features and initial features
            r = (1 - alpha) * hi + alpha * h0  # Mix neighbor info and original features
        else:
            support = (1 - alpha) * hi + alpha * h0  # Mix neighbor info and original features
            r = support  # Without variant, support equals r
        
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, dia_len):
        """
        x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        tmpx = torch.zeros(0).cuda()
        tmp = 0
        for i in dia_len:
            a = x[tmp:tmp+i].unsqueeze(1)
            a = a + self.pe[:a.size(0)]
            tmpx = torch.cat([tmpx, a], dim=0)
            tmp = tmp + i
        return self.dropout(tmpx)


class GCN(nn.Module):
    def __init__(self, n_dim, nhidden, dropout, lamda, alpha, variant, return_feature, use_residue, 
                new_graph='full', n_speakers=2, modals=['a', 'v', 'l'], use_speaker=True, use_modal=False, num_L=3, num_K=4):
        super(GCN, self).__init__()
        self.return_feature = return_feature  # Whether to return features
        self.use_residue = use_residue
        self.new_graph = new_graph  # Whether to use new graph structure

        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

        # Multimodal input configuration: audio, video, language
        self.modals = modals
        self.modal_embeddings = nn.Embedding(3, n_dim)  # Embedding for modalities
        self.speaker_embeddings = nn.Embedding(n_speakers, n_dim)  # Embedding for speakers
        self.use_speaker = use_speaker  # Whether to use speaker embedding
        self.use_modal = use_modal  # Whether to use modality embedding
        self.use_position = False  # Position encoding currently not used

        # Fully connected layer in network
        self.fc1 = nn.Linear(n_dim, nhidden)  # Mapping from input dimension to hidden layer

        # Graph convolution parameters
        self.num_L = num_L
        self.num_K = num_K
        self.act_fn = nn.ReLU()
        self.hyperedge_weight = nn.Parameter(torch.ones(1000))  # Hyperedge weights
        self.EW_weight = nn.Parameter(torch.ones(5200))
        self.hyperedge_attr1 = nn.Parameter(torch.rand(nhidden))  # Hyperedge feature 1
        self.hyperedge_attr2 = nn.Parameter(torch.rand(nhidden))  # Hyperedge feature 2
        # Graph convolution layers
        for kk in range(num_K):
            setattr(self, 'conv%d' % (kk + 1), GraphGCN(nhidden, nhidden))

    def forward(self, a, v, l, dia_len, qmask, epoch):
        qmask = torch.cat([qmask[:x, i, :] for i, x in enumerate(dia_len)], dim=0)
        spk_idx = torch.argmax(qmask, dim=-1)  # Get speaker indices
        spk_emb_vector = self.speaker_embeddings(spk_idx)  # Get speaker embedding vectors
        # Add speaker embeddings to language modality if enabled
        if self.use_speaker:
            if 'l' in self.modals:
                l += spk_emb_vector
        # Add positional encoding to each modality if enabled
        if self.use_position:
            if 'l' in self.modals:
                l = self.l_pos(l, dia_len)
            if 'a' in self.modals:
                a = self.a_pos(a, dia_len)
            if 'v' in self.modals:
                v = self.v_pos(v, dia_len)
        

        # Create adjacency matrix and features needed for graph convolution
        gnn_edge_index, gnn_features = self.create_gnn_index(a, v, l, dia_len, self.modals)
        x1 = self.fc1(gnn_features)
        out = x1
        gnn_out = x1
        # Apply multiple graph convolution layers
        for kk in range(self.num_K):
            gnn_out = gnn_out + getattr(self, 'conv%d' % (kk + 1))(gnn_out, gnn_edge_index)
        # Concatenate convolution result with initial output
        out2 = torch.cat([out, gnn_out], dim=1)
        if self.use_residue:
            out2 = torch.cat([gnn_features, out2], dim=-1)
        # Reverse features and return
        out1 = self.reverse_features(dia_len, out2)

        return out1

    def reverse_features(self, dia_len, features):
        # Split features according to dialogue lengths and reorder
        l = []
        a = []
        v = []
        for i in dia_len:
            ll = features[0:1 * i]
            aa = features[1 * i:2 * i]
            vv = features[2 * i:3 * i]
            features = features[3 * i:]
            l.append(ll)
            a.append(aa)
            v.append(vv)
        tmpl = torch.cat(l, dim=0)
        tmpa = torch.cat(a, dim=0)
        tmpv = torch.cat(v, dim=0)
        features = torch.cat([tmpl, tmpa, tmpv], dim=-1)
        return features

    def create_gnn_index(self, a, v, l, dia_len, modals):
        # Create adjacency matrix (edge indices) needed for graph convolution
        num_modality = len(modals)
        node_count = 0
        index = []
        tmp = []
        
        for i in dia_len:
            nodes = list(range(i * num_modality))  # Create nodes for each modality
            nodes = [j + node_count for j in nodes]
            nodes_l = nodes[0:i * num_modality // 3]
            nodes_a = nodes[i * num_modality // 3:i * num_modality * 2 // 3]
            nodes_v = nodes[i * num_modality * 2 // 3:]
            index = index + list(permutations(nodes_l, 2)) + list(permutations(nodes_a, 2)) + list(permutations(nodes_v, 2))
            # Create hyperedges connecting different modality nodes
            Gnodes = []
            for _ in range(i):
                Gnodes.append([nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]])
            for ii, _ in enumerate(Gnodes):
                tmp = tmp + list(permutations(_, 2))
            # Concatenate features
            if node_count == 0:
                ll = l[0:0 + i]
                aa = a[0:0 + i]
                vv = v[0:0 + i]
                features = torch.cat([ll, aa, vv], dim=0)
                temp = 0 + i
            else:
                ll = l[temp:temp + i]
                aa = a[temp:temp + i]
                vv = v[temp:temp + i]
                features_temp = torch.cat([ll, aa, vv], dim=0)
                features = torch.cat([features, features_temp], dim=0)
                temp = temp + i
            node_count = node_count + i * num_modality
        edge_index = torch.cat([torch.LongTensor(index).T, torch.LongTensor(tmp).T], 1).to("cuda:0")
        return edge_index, features
