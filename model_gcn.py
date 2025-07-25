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



