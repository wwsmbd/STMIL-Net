import math
import torch
from models.graph_supcon import UnifiedContrastiveModel
from utils.graph_utils import create_fully_connected_graph, EarlyStopping
from utils.losses import combined_supcon_loss

# Hyperparameters
temperature = 0.07
hidden_channels = 64
out_channels = 32
feature_dim = 128
sequence_length = X_train_tensor.shape[2]

# Sliding window configuration
window_size = math.ceil(sequence_length / 20)
stride = math.ceil(window_size / 4)

# Model initialization
model = UnifiedContrastiveModel(
    feature_extractor=inception_feature_extractor,
    spatial_hidden_dim=hidden_channels,
    temporal_hidden_dim=hidden_channels,
    out_channels=out_channels,
    feature_dim=feature_dim,
    window_size=window_size,
    stride=stride,
    reduction_method="mean"
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
early_stopping = EarlyStopping(patience=20, verbose=True)
best_val_loss = float("inf")
best_model_state = None

# Training loop
n_epochs = 1000
for epoch in range(n_epochs):
    model.train()
    train_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        spatial_edge_index = create_fully_connected_graph(x.mean(dim=-1)).to(device)
        num_windows = (x.shape[2] - window_size) // stride + 1
        temporal_edge_index = model.create_temporal_edge_index(num_windows).to(device)

        spatial_proj, temporal_proj = model(x, spatial_edge_index, temporal_edge_index, y)

        loss = combined_supcon_loss(
            spatial_features=spatial_proj,
            temporal_features=temporal_proj,
            labels=y,
            temperature=temperature
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            spatial_edge_index = create_fully_connected_graph(x.mean(dim=-1)).to(device)
            num_windows = (x.shape[2] - window_size) // stride + 1
            temporal_edge_index = model.create_temporal_edge_index(num_windows).to(device)

            spatial_proj, temporal_proj = model(x, spatial_edge_index, temporal_edge_index, y)

            loss = combined_supcon_loss(
                spatial_features=spatial_proj,
                temporal_features=temporal_proj,
                labels=y,
                temperature=temperature
            )
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch + 1}")
        break

# Load best model
model.load_state_dict(best_model_state)

# Test evaluation
model.eval()
test_loss = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        spatial_edge_index = create_fully_connected_graph(x.mean(dim=-1)).to(device)
        num_windows = (x.shape[2] - window_size) // stride + 1
        temporal_edge_index = model.create_temporal_edge_index(num_windows).to(device)

        spatial_proj, temporal_proj = model(x, spatial_edge_index, temporal_edge_index, y)

        loss = combined_supcon_loss(
            spatial_features=spatial_proj,
            temporal_features=temporal_proj,
            labels=y,
            temperature=temperature
        )
        test_loss += loss.item()
test_loss /= len(test_loader)

print(f"Test Loss: {test_loss:.4f}")
