from torch import nn
import torch

class GRU(nn.Module):
    def __init__(self,d_feat=158,hidden_size=256,num_layers=2,dropout=0.0,gate_input_start_index=158):
        super().__init__()

        self.rnn = nn.GRU(
            input_size = d_feat,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout,
        )
        self.fc_out = nn.Linear(hidden_size,1)
        self.d_feat = d_feat
        self.gate_input_start_index = gate_input_start_index
    
    def forward(self,x):
        src = x[:, :, :self.gate_input_start_index]

        out, _ = self.rnn(src)
        return self.fc_out(out[:, -1, :]).squeeze()