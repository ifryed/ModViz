"""A dependency-free (except PyTorch) hybrid CNN + GNN example.

Each graph node owns one RGB image. The CNN turns each image into a feature
vector, then two dense graph-convolution layers exchange information between
connected nodes and produce a class prediction for every node.
"""

import torch
from torch import nn


class DenseGraphConv(nn.Module):
    """Simple GCN layer using a dense [nodes, nodes] adjacency matrix."""

    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.projection = nn.Linear(input_features, output_features)

    def forward(self, features, adjacency):
        # Add self-loops and symmetrically normalize the adjacency matrix.
        adjacency = adjacency + torch.eye(
            adjacency.size(0), device=adjacency.device, dtype=adjacency.dtype
        )
        degree = adjacency.sum(dim=1).clamp(min=1)
        scale = degree.rsqrt()
        normalized = scale[:, None] * adjacency * scale[None, :]
        return self.projection(normalized @ features)


class HybridCNNGNN(nn.Module):
    def __init__(self, classes: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.graph1 = DenseGraphConv(32, 24)
        self.graph_relu = nn.ReLU()
        self.graph2 = DenseGraphConv(24, classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, node_images, adjacency, targets):
        # node_images: [number_of_nodes, 3, height, width]
        features = self.pool(self.relu1(self.conv1(node_images)))
        features = self.global_pool(self.relu2(self.conv2(features)))
        features = torch.flatten(features, 1)
        features = self.graph_relu(self.graph1(features, adjacency))
        logits = self.graph2(features, adjacency)
        loss = self.loss_fn(logits, targets)
        return logits, loss


if __name__ == "__main__":
    model = HybridCNNGNN()
    images = torch.zeros(8, 3, 32, 32)
    adjacency = torch.eye(8)
    targets = torch.zeros(8, dtype=torch.long)
    logits, loss = model(images, adjacency, targets)
    print(logits.shape, loss.shape)  # [8, 4], scalar

