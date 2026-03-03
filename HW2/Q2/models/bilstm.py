import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size = 64, num_layers = 2, dropout= 0.0, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size*4, hidden_size*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )


    def forward(self, x, seq_mask=None, **kwargs):
        batch_size, seq_len, _ = x.size()

        if seq_mask is not None:
            lengths = seq_mask.sum(dim=1).long() 
            packed_input = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed_input)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=seq_len
            )
        else:
            out, _ = self.lstm(x)

        if seq_mask is not None:
            mask_exp = seq_mask.unsqueeze(-1)  
            mask_exp = mask_exp.expand(-1, -1, out.size(2))  

            out_masked_max = out.clone()
            out_masked_max[mask_exp == 0] = float('-inf')
            max_pool, _ = torch.max(out_masked_max, dim=1)

            sum_pool = (out * mask_exp).sum(dim=1)
            lengths_clamped = lengths.clamp(min=1).unsqueeze(1)  
            avg_pool = sum_pool / lengths_clamped.float()
        else:
            max_pool, _ = torch.max(out, dim=1)
            avg_pool = torch.mean(out, dim=1)

        features = torch.cat([max_pool, avg_pool], dim=1)
        preds = self.mlp(features)
        return preds.squeeze(-1)


