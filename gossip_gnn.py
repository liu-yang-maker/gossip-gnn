import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data
import random

class GNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GNNLayer, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Linear transformation.
        x = self.lin(x)

        # Start propagating messages.
        return self.propagate(edge_index, x=x)

    def message(self, x_j, edge_index_i, size):
        # x_j corresponds to the source node features in edge_index.
        # Introduce gossip mechanism: randomly select a neighbor's message with some probability.
        if random.random() < 0.5:
            neighbor_idx = random.randint(0, edge_index_i.size(0) - 1)
            return x_j[neighbor_idx]
        else:
            return x_j

    def update(self, aggr_out):
        # Return new node embeddings.
        return aggr_out

# Example usage
if __name__ == "__main__":
    # Create a simple graph.
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)  # Shape [2, num_edges]
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)  # Node feature matrix with 3 nodes, each with feature size 1

    # Create the GNN layer.
    gnn_layer = GNNLayer(in_channels=1, out_channels=2)

    # Forward pass.
    out = gnn_layer(x, edge_index)
    print(out)

    # Define a simple graph data object (optional, for more complex cases).
    data = Data(x=x, edge_index=edge_index)

    # You can stack multiple GNN layers to build a deeper network.
    gnn_layer2 = GNNLayer(in_channels=2, out_channels=2)
    out = gnn_layer2(out, edge_index)
    print(out)