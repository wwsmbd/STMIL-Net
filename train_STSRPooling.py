import torch
import torch.nn as nn

from STSRPooling import STSRPooling  # Assuming STSRPooling is defined in this file


def set_embedding_trainable(model, trainable=True, part="inception"):
    """
    Set part of the model to be trainable or frozen.
    Args:
        model: Model that includes a feature_extractor.
        trainable (bool): If True, set to train mode. If False, freeze the parameters.
        part (str): Target part name (currently only "inception" supported).
    """
    if part == "inception":
        for param in model.feature_extractor.parameters():
            param.requires_grad = trainable
        if trainable:
            model.feature_extractor.train()
        else:
            model.feature_extractor.eval()
    else:
        raise ValueError(f"Unknown part: {part}. Only 'inception' is supported.")


# Get sample input to determine output channel size
data_sample = next(iter(train_loader))[0].to(device)
with torch.no_grad():
    sample_output = model.feature_extractor(data_sample)
    print(f"Inception Feature Extractor Output Shape: {sample_output.shape}")
    d_in = sample_output.size(1)  # channel size

# Initialize STSR Pooling module
n_clz = len(torch.unique(y_train_tensor))
stsr_pooling = STSRPooling(
    d_in=d_in,
    n_clz=n_clz,
    d_attn=8,
    reduction_ratio=4
).to(device)

# Optimizer, Scheduler, Criterion
mil_optimizer = torch.optim.Adam(stsr_pooling.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(mil_optimizer, mode='min', factor=0.5, patience=10, verbose=True)
criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

# Freeze feature extractor
set_embedding_trainable(model, trainable=False, part="inception")
n_epochs = 200
best_val_loss = float('inf')

# Training loop
for epoch in range(n_epochs):
    model.feature_extractor.eval()
    stsr_pooling.train()
    train_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            embeddings = model.feature_extractor(x)

        mil_optimizer.zero_grad()
        output = stsr_pooling(embeddings)
        logits = output["bag_logits"]
        loss = criterion(logits, y)
        loss.backward()
        mil_optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validation loop
    model.feature_extractor.eval()
    stsr_pooling.eval()
    val_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            embeddings = model.feature_extractor(x)
            output = stsr_pooling(embeddings)
            logits = output["bag_logits"]
            loss = criterion(logits, y)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    scheduler.step(val_loss)
    print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_mil_state = stsr_pooling.state_dict()

# Load best model
stsr_pooling.load_state_dict(best_mil_state)

# Test evaluation
stsr_pooling.eval()
test_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        embeddings = model.feature_extractor(x)
        output = stsr_pooling(embeddings)
        logits = output["bag_logits"]
        loss = criterion(logits, y)
        test_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

test_loss /= len(test_loader)
accuracy = correct / total
print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

# Optional: Visualize attention outputs
time_attention = output["time_attention"]
channel_attention = output["channel_attention"]
tsr_weight = output["tsr_weight"]
print(f"Time Attention Shape: {time_attention.shape}")
print(f"Channel Attention Shape: {channel_attention.shape}")
print(f"TSR Weight Shape: {tsr_weight.shape}")
