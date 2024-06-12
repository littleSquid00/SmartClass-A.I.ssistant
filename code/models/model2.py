import torch
import torch.nn as nn
import torch.nn.functional as F


class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, padding=1
            ),  # Shape: (48, 48, 16)
            nn.BatchNorm2d(16),  # Batch Normalization layer
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2),  # Shape: (24, 24, 16)
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, padding=1
            ),  # Shape: (24, 24, 32)
            nn.BatchNorm2d(32),  # Batch Normalization layer
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2),  # Shape: (12, 12, 32)
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, padding=1
            ),  # Shape: (12, 12, 64)
            nn.BatchNorm2d(64),  # Batch Normalization layer
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2),  # Shape: (6, 6, 64)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(
                in_features=6 * 6 * 64, out_features=1000
            ),  # Adjust to match the output of the conv layers
            nn.LeakyReLU(),
            nn.Linear(in_features=1000, out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=4),
        )

    def forward(self, x):
        out = self.conv_layer(x)
        flattened_out = torch.flatten(out, 1)
        out = self.fc_layer(flattened_out)
        out = F.log_softmax(out, dim=1)
        return out
