import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGEEncoder(nn.Module):
    """GraphSAGE-based encoder."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Args:
            x: Node features, shape (num_nodes, in_channels)
            edge_index: Edge index of the graph
        Returns:
            Node embeddings, shape (num_nodes, out_channels)
        """
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.normalize(x, dim=1)


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning."""
    def __init__(self, in_features, out_features=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )

    def forward(self, x):
        return F.normalize(self.fc(x), dim=1)


class UnifiedContrastiveModel(nn.Module):
    """
    Unified model for spatial and temporal graph-based contrastive learning.
    """
    def __init__(self, feature_extractor, spatial_hidden_dim, temporal_hidden_dim, out_channels, feature_dim=128, window_size=10, stride=5, reduction_method="mean"):
        super().__init__()
        self.feature_extractor = feature_extractor

        self.spatial_graph_encoder = GraphSAGEEncoder(
            in_channels=out_channels * 4,
            hidden_channels=spatial_hidden_dim,
            out_channels=out_channels
        )
        self.spatial_projection_head = ProjectionHead(
            in_features=out_channels,
            out_features=feature_dim
        )

        self.temporal_graph_encoder = GraphSAGEEncoder(
            in_channels=out_channels * 4,
            hidden_channels=temporal_hidden_dim,
            out_channels=out_channels
        )
        self.temporal_projection_head = ProjectionHead(
            in_features=out_channels,
            out_features=feature_dim
        )

        self.window_size = window_size
        self.stride = stride
        self.reduction_method = reduction_method

    def sliding_window_with_reduction(self, data):
        """
        Apply sliding window and reduce data within each window.

        Args:
            data (Tensor): Shape [batch, channels, sequence_length]
        Returns:
            Tensor: Reduced window embeddings, shape [batch, num_windows, channels]
        """
        batch_size, channels, seq_length = data.shape
        windows = []

        for i in range(0, seq_length - self.window_size + 1, self.stride):
            windows.append(data[:, :, i:i + self.window_size])

        windows = torch.stack(windows, dim=1)  # [batch, num_windows, channels, window_size]

        if self.reduction_method == "mean":
            return windows.mean(dim=-1)  # [batch, num_windows, channels]
        elif self.reduction_method == "mlp":
            mlp = nn.Linear(self.window_size, 1).to(data.device)
            return mlp(windows).squeeze(-1)

        return windows

    def create_temporal_edge_index(self, num_windows):
        """
        Create edge index for fully-connected temporal graph.

        Args:
            num_windows (int): Number of windows (nodes)
        Returns:
            edge_index (Tensor): Shape [2, num_edges]
        """
        edge_index = torch.combinations(torch.arange(num_windows), r=2).T
        return edge_index

    def forward(self, x, spatial_edge_index, temporal_edge_index, labels=None):
        features = self.feature_extractor(x)  # [batch, channels, time]

        spatial_embeddings = self.spatial_graph_encoder(features.mean(dim=-1), spatial_edge_index)
        spatial_projections = self.spatial_projection_head(spatial_embeddings)

        reduced_windows = self.sliding_window_with_reduction(features)  # [batch, num_windows, channels]
        batch_size, num_windows, channels = reduced_windows.shape
        reduced_windows_flat = reduced_windows.view(batch_size * num_windows, channels)

        temporal_embeddings = self.temporal_graph_encoder(reduced_windows_flat, temporal_edge_index)
        temporal_embeddings = temporal_embeddings.view(batch_size, num_windows, -1).mean(dim=1)
        temporal_projections = self.temporal_projection_head(temporal_embeddings)

        return spatial_projections, temporal_projections


def create_fully_connected_graph(data):
    """
    Create fully connected graph from given data.

    Args:
        data (Tensor): Node features of shape [num_nodes, features]
    Returns:
        edge_index (Tensor): Fully connected edge index [2, num_edges]
    """
    num_nodes = data.size(0)
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).T
    return edge_index.to(torch.int64)


class EarlyStopping:
    """Early stopping utility based on validation loss."""
    def __init__(self, patience=20, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def supervised_contrastive_loss(features, labels, temperature=0.07):
    """
    Compute Supervised Contrastive Loss.

    Args:
        features (Tensor): Latent features, shape [batch_size, feature_dim]
        labels (Tensor): Class labels, shape [batch_size]
        temperature (float): Temperature parameter
    Returns:
        Tensor: Scalar loss value
    """
    labels = labels.unsqueeze(1)
    mask = torch.eq(labels, labels.T).float()

    features = F.normalize(features, dim=1)
    similarity = torch.matmul(features, features.T) / temperature

    logits_max, _ = similarity.max(dim=1, keepdim=True)
    logits = similarity - logits_max.detach()
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

    loss = -(mask * log_prob).sum(dim=1) / mask.sum(dim=1)
    return loss.mean()


def combined_supcon_loss(spatial_features, temporal_features, labels, temperature=0.07):
    """
    Compute combined contrastive loss from spatial and temporal features.

    Args:
        spatial_features (Tensor): [batch_size, feature_dim]
        temporal_features (Tensor): [batch_size, feature_dim]
        labels (Tensor): [batch_size]
        temperature (float): Temperature parameter
    Returns:
        Tensor: Total loss
    """
    spatial_loss = supervised_contrastive_loss(spatial_features, labels, temperature)
    temporal_loss = supervised_contrastive_loss(temporal_features, labels, temperature)
    return spatial_loss + temporal_loss
