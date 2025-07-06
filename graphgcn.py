import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


def generate_random_mask(num_nodes, num_features, mask_prob=0.5):
    """
    Randomly generate a mask matrix. Each feature is masked based on the given probability.
    :param num_nodes: Number of nodes
    :param num_features: Feature dimension of each node
    :param mask_prob: Masking probability (between 0 and 1); higher values result in more masked features.
    :return: Boolean mask matrix
    """
    mask = torch.rand((num_nodes, num_features)) > mask_prob  # Generate a boolean mask
    return mask


class GraphGCN(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add'):
        # Set aggregation method to 'add'
        super(GraphGCN, self).__init__(aggr='add')  # "Add" aggregation.
        # self.lin = torch.nn.Linear(in_channels, out_channels)

        # Gate layer: concatenate features of node pairs and pass them through a linear layer
        # Input size is 2 * in_channels (due to concatenation), output size is 1 for gating
        self.gate = torch.nn.Linear(2 * in_channels, 1)

    def forward(self, x, edge_index):
        # Get number of nodes and feature dimensions
        num_nodes, dim = x.shape
        # Generate a random mask to prevent some features from being used during training (similar to Dropout)
        mask = generate_random_mask(num_nodes, dim, mask_prob=0.5)
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: (Optional) Add self-loops to adjacency matrix
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: (Optional) Apply linear transformation to node features
        # x = self.lin(x)

        # Step 3-5: Start propagating messages
        # Call propagate to perform message passing on the graph; x * mask applies the mask
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x * mask)

    def message(self, x_i, x_j, edge_index, size):
        # x_i and x_j are the features of the target and source nodes, respectively
        # edge_index defines edge connections in the graph
        # x_j shape: [num_edges, feature_dim]

        # Step 3: Normalize node features by computing node degree
        row, col = edge_index  # row: source node, col: target node
        deg = degree(row, size[0], dtype=x_j.dtype)  # Compute source node degrees
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # Compute normalization factor

        # Concatenate target and source node features, and pass through gate layer to compute gate value
        h2 = torch.cat([x_i, x_j], dim=1)
        alpha_g = torch.tanh(self.gate(h2))  # Compute gate values using tanh activation

        # Normalize and modulate messages using gate value
        return norm.view(-1, 1) * x_j * alpha_g

    
