"""Chapter 4: architecture-only versions of the chapter 5 RNN/Transformer.

No simulator, training loop or checkpoints. Classes retain the chapter 5 structure.
"""
import torch
from torch import nn

DOF = 6
HISTORY = 10

class VanillaRNN_BC(nn.Module):
    def __init__(self, dof=DOF, embed_dim=64, hidden_dim=64):
        super().__init__()
        self.projector = nn.Linear(dof, embed_dim)
        self.rnn = nn.RNN(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, dof)

    def forward(self, x):
        embedded = torch.relu(self.projector(x))
        _, hidden = self.rnn(embedded)
        pred_action = self.head(hidden[-1])
        return pred_action

class TransformerEncoderBC(nn.Module):
    def __init__(self, dof=DOF, history=HISTORY, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.state_projector = nn.Linear(dof, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, history, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, dof)

    def forward(self, state_seq):
        x = self.state_projector(state_seq) + self.pos_embedding
        x = self.transformer_encoder(x)
        return self.head(x[:, -1])
