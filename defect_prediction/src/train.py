import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

def make_loader(X, y, batch_size=32, shuffle=True):
    """Convert numpy arrays to a PyTorch DataLoader."""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(model, X_train, y_train, X_val, y_val,
                epochs=50, lr=1e-3, batch_size=32, device='cpu',
                output_dir='outputs'):
    """
    Train the model with early stopping (patience=10).
    Returns history dict with train/val loss per epoch.
    Saves trained model to output_dir/model.pt.
    """
    model.to(device)

    # Weighted BCE to further handle any residual imbalance
    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
        dtype=torch.float32
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   batch_size, shuffle=False)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = None

    for epoch in range(1, epochs + 1):
        # ── Training
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        scheduler.step(val_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # ── Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch}")
                break

    # Restore best weights
    if best_weights:
        model.load_state_dict(best_weights)

    # ── Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = f"{output_dir}/model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved → {model_path}")

    return history
