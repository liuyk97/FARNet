from torch import nn


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv = nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU()
        self.conv_out = nn.Conv2d(inter_channels, channels, 1)

    def forward(self, input):
        out = self.conv(input)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv_out(out)
        return out