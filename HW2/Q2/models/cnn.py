import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, input_channels=4, base_filters=32, out_dim=1, dropout_prob=0.0):
        super().__init__()

        self.conv1_3 = nn.Conv1d(input_channels, base_filters, 3, padding=1)
        self.bn1_3 = nn.BatchNorm1d(base_filters)

        self.conv1_5 = nn.Conv1d(input_channels, base_filters, 5, padding=2)
        self.bn1_5 = nn.BatchNorm1d(base_filters)

        self.conv1_7 = nn.Conv1d(input_channels, base_filters, 7, padding=3)
        self.bn1_7 = nn.BatchNorm1d(base_filters)


        self.conv2 = nn.Conv1d(base_filters * 3, base_filters * 2, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(base_filters * 2)


        self.dropout = nn.Dropout(dropout_prob)


        self.fc1 = nn.Linear(base_filters * 2, 32)
        self.fc2 = nn.Linear(32, out_dim)

    def forward(self, x, seq_mask=None):
        x = x.permute(0, 2, 1)

        x3 = F.relu(self.bn1_3(self.conv1_3(x)))
        x5 = F.relu(self.bn1_5(self.conv1_5(x)))
        x7 = F.relu(self.bn1_7(self.conv1_7(x)))

        x = torch.cat([x3, x5, x7], dim=1)
        x = F.relu(self.bn2(self.conv2(x)))


        if seq_mask is not None:
            mask_exp = seq_mask.unsqueeze(1).expand(-1, x.size(1), -1)

            x_masked = x.clone()
            x_masked[mask_exp == 0] = float('-inf')
            x_max, _ = torch.max(x_masked, dim=2)

            x_sum = (x * mask_exp.float()).sum(dim=2)
            lengths = seq_mask.sum(dim=1).clamp(min=1).unsqueeze(1)
            x_avg = x_sum / lengths
        else:
            x_max = F.adaptive_max_pool1d(x, 1).squeeze(-1)
            x_avg = F.adaptive_avg_pool1d(x, 1).squeeze(-1)

        x = x_max + x_avg

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x.squeeze(-1)