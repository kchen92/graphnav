import torch
import torch.nn as nn
import torch.nn.functional as F


class BehaviorCNN(nn.Module):

    def __init__(self, in_channels):
        super(BehaviorCNN, self).__init__()
        self.is_recurrent = False

        # Input resolution: 320x240
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=2, padding=0)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=0)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0)
        self.conv5_bn = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0)
        self.conv6_bn = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=(2, 3), stride=1, padding=0)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = torch.tanh(self.conv7(x))
        x = torch.squeeze(x, dim=3)
        x = torch.squeeze(x, dim=2)
        return x
