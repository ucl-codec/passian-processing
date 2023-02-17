''' Implementation of the Residual Attention Model (one of SOTA for Alz classificaiton)?'''
import torch
import torch.nn as nn

class ResidualAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualAttentionBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.attention = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention(out) * out
        out += residual
        out = self.relu(out)
        return out

class ResidualAttentionNetwork(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResidualAttentionNetwork, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.rab1 = ResidualAttentionBlock(64, 128)
        self.rab2 = ResidualAttentionBlock(128, 256)
        self.rab3 = ResidualAttentionBlock(256, 512)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.rab1(out)
        out = self.rab2(out)
        out = self.rab3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

