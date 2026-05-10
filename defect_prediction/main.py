"""
main.py — Run the full pipeline:
  1. Load & preprocess data
  2. Build LSTM model
  3. Train with early stopping
  4. Evaluate & save results
"""

import torch
import os
from src.preprocess import load_and_preprocess
from src.model      import DefectLSTM
from src.train      import train_model
from src.evaluate   import evaluate_and_save

# ── Config
DATA_PATH  = 'data/kc1.csv'
OUTPUT_DIR = 'outputs'
EPOCHS     = 80
LR         = 1e-3
BATCH_SIZE = 32
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {DEVICE}")

# ── 1. Preprocess
X_train, X_val, X_test, y_train, y_val, y_test, scaler = \
    load_and_preprocess(DATA_PATH)

input_size = X_train.shape[1]
print(f"Input features: {input_size}")

# ── 2. Build Model
model = DefectLSTM(
    input_size=input_size,
    hidden_size=64,
    num_layers=2,
    dropout=0.3
)
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")

# ── 3. Train
print("\n── Training ────────────────────────────")
history = train_model(
    model, X_train, y_train, X_val, y_val,
    epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, device=DEVICE
)

# ── 4. Save model
os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.save(model.state_dict(), f"{OUTPUT_DIR}/model.pt")
print(f"Model saved → {OUTPUT_DIR}/model.pt")

# ── 5. Evaluate
print("\n── Evaluation ──────────────────────────")
metrics = evaluate_and_save(model, X_test, y_test, history,
                             output_dir=OUTPUT_DIR, device=DEVICE)