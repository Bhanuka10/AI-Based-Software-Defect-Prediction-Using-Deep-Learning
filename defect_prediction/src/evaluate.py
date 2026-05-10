import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

def predict(model, X, device='cpu', threshold=0.5):
    """Run inference and return binary predictions + probabilities."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_t)
        probs  = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= threshold).astype(int)
    return preds, probs


def evaluate_and_save(model, X_test, y_test, history,
                      output_dir='outputs', device='cpu'):
    """
    Compute all metrics, print a report, save metrics.json,
    save the model to model.pt, and save confusion matrix + ROC curve + training curves.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)

    preds, probs = predict(model, X_test, device)

    # ── Metrics
    metrics = {
        'accuracy':  round(accuracy_score(y_test, preds),  4),
        'f1_score':  round(f1_score(y_test, preds),        4),
        'precision': round(precision_score(y_test, preds), 4),
        'recall':    round(recall_score(y_test, preds),    4),
        'roc_auc':   round(roc_auc_score(y_test, probs),   4),
    }
    print("\n── Evaluation Results ──────────────────")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v}")
    print("\n── Classification Report ───────────────")
    print(classification_report(y_test, preds, target_names=['Clean', 'Defective']))

    # ── Save metrics
    with open(f"{output_dir}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved → {output_dir}/metrics.json")

    # ── Save model
    torch.save(model.state_dict(), f"{output_dir}/model.pt")
    print(f"Model saved → {output_dir}/model.pt")

    # ── Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Clean', 'Defective'],
                yticklabels=['Clean', 'Defective'], ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    fig.tight_layout()
    fig.savefig(f"{output_dir}/plots/confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ── ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve'); ax.legend()
    fig.tight_layout()
    fig.savefig(f"{output_dir}/plots/roc_curve.png", dpi=150)
    plt.close(fig)

    # ── Training Curves
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history['train_loss'], label='Train Loss')
    ax.plot(history['val_loss'],   label='Val Loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss'); ax.legend()
    fig.tight_layout()
    fig.savefig(f"{output_dir}/plots/training_curves.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved → {output_dir}/plots/")
    return metrics
