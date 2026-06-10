import torch
import torch.nn as nn

class ExpectedPointsLSTM(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=256, output_dim=1, num_layers=2, dropout=0.2):
        super(ExpectedPointsLSTM, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU() 
        
    def forward(self, x, lengths):
        self.lstm.flatten_parameters()
        x = self.layer_norm(x)
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed_x)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=400
        ) 
        predictions = self.relu(self.fc(out))
        return predictions