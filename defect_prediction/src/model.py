import torch
import torch.nn as nn

class DefectLSTM(nn.Module):
    """
    LSTM-based classifier for tabular software metrics.

    Architecture:
      Input (batch, features) → unsqueeze as sequence of length 1
      → LSTM (hidden_size=64, 2 layers) → last hidden state
      → Dropout → Linear → logit output
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(DefectLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,          # (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)           # binary output (logit)
        )

    def forward(self, x):
        # x shape: (batch, features)
        # Add sequence dimension: (batch, 1, features)
        x = x.unsqueeze(1)

        # LSTM: output shape (batch, 1, hidden), h_n shape (num_layers, batch, hidden)
        _, (h_n, _) = self.lstm(x)

        # Take last layer's hidden state
        last_hidden = h_n[-1]           # (batch, hidden_size)

        logits = self.classifier(last_hidden)   # (batch, 1)
        return logits.squeeze(1)        # (batch,)
