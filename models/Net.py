import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter
from typing import List
from models import resnet
from models.SegHead import *

classifier_map = {
    'fcn': FCNHead,
    'unet': UNet,
}


class CMR_Module(nn.Module):
    def __init__(self, channels=1):
        super(CMR_Module, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class MFDS_Module(nn.Module):
    def __init__(self, in_channel):
        super(MFDS_Module, self).__init__()
        self.conv12 = nn.Conv2d(in_channels=in_channel * 2, out_channels=in_channel, kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.conv23 = nn.Conv2d(in_channels=in_channel * 2, out_channels=in_channel, kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.conv34 = nn.Conv2d(in_channels=in_channel * 2, out_channels=in_channel, kernel_size=1, stride=1, padding=0,
                                bias=False)

    def forward(self, f_1, f_2, f_3, f_4):
        f_12 = torch.cat([f_1, f_2], dim=1)
        f_12 = self.conv12(f_12)

        f_123 = torch.cat([f_12, f_3], dim=1)
        f_123 = self.conv23(f_123)

        f_1234 = torch.cat([f_123, f_4], dim=1)
        f_1234 = self.conv34(f_1234)

        return f_1234


class CFI_Module(nn.Module):
    def __init__(self, in_channel, out_channel=256):
        super(CFI_Module, self).__init__()
        self.inter_channel = in_channel // 2
        self.conv_phi = nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                  padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=in_channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                    padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=in_channel * 2, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                padding=0, bias=False)
        self.conv_AB = nn.Conv2d(in_channels=in_channel * 2, out_channels=out_channel, kernel_size=1, stride=1,
                                 padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=out_channel, kernel_size=1, stride=1,
                                   padding=0, bias=False)

    def forward(self, A, B):
        # [N, C, H , W]
        b, c, h, w = A.size()
        AB = torch.cat([A, B], dim=1)
        # [N, C/2, H * W]
        x_phi = self.conv_phi(A).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(B).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(AB).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        # [N, C, H , W]
        out = self.conv_mask(mul_theta_phi_g)
        AB = self.conv_AB(AB)
        out = AB + out
        return out


class ChangeDetectionModel(nn.Module):

    def __init__(self, dilation: List[bool], backbone=None, channel=256, out_channel=64, CFI=False, MFDS=False,
                 CMR=False, ):
        super(ChangeDetectionModel, self).__init__()
        self.backbone = backbone
        self.classifier = FCNHead(out_channel, 2)
        self.CFI = CFI
        self.MFDS = MFDS
        self.CMR = CMR
        self.dilation = dilation
        if self.CFI:  # Cross-Temporal Feature Interaction Module
            self.CFI_1 = CFI_Module(in_channel=channel, out_channel=out_channel)
            self.CFI_2 = CFI_Module(in_channel=channel, out_channel=out_channel)
            self.CFI_3 = CFI_Module(in_channel=channel, out_channel=out_channel)
            self.CFI_4 = CFI_Module(in_channel=channel, out_channel=out_channel)
        else:
            self.CFI_1 = nn.Conv2d(channel * 2, out_channel, kernel_size=1, bias=False)
            self.CFI_2 = nn.Conv2d(channel * 2, out_channel, kernel_size=1, bias=False)
            self.CFI_3 = nn.Conv2d(channel * 2, out_channel, kernel_size=1, bias=False)
            self.CFI_4 = nn.Conv2d(channel * 2, out_channel, kernel_size=1, bias=False)

        self.conv1 = nn.Conv2d(256, channel, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(512, channel, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(1024, channel, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(2048, channel, kernel_size=1, bias=False)
        if self.MFDS:  # Multi-scale Feature Supplement Module
            self.MFDSM = MFDS_Module(in_channel=out_channel)
        if self.CMR:  # Change Map Restoration Module
            self.CMRM = CMR_Module(channels=2)
            self.conv_out = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1, padding=0, bias=False)

    # def forward(self, input):
    #     A = input[:, :3, :, :]
    #     B = input[:, 3:6, :, :]
    def forward(self, A, B):
        input_shape = A.shape[-2:]
        features_A = self.backbone(A)
        features_B = self.backbone(B)
        f_A1 = features_A['layer1']
        f_A2 = features_A['layer2']
        f_A3 = features_A['layer3']
        f_A4 = features_A['layer4']
        f_B1 = features_B['layer1']
        f_B2 = features_B['layer2']
        f_B3 = features_B['layer3']
        f_B4 = features_B['layer4']
        # for i, flag in enumerate(self.dilation): index = i+2 if flag is False: loc = {} from torch.nn import
        # functional as F exec(f"f_A{index} = F.interpolate(f_A{index}, size=f_A1.shape[-2:], mode='bilinear',
        # align_corners=False)", locals(), loc) exec(f"f_B{index} = F.interpolate(f_B{index}, size=f_A1.shape[-2:],
        # mode='bilinear', align_corners=False)", locals(), loc) globals().update(loc)
        if f_A2.shape[-2:] != f_A1.shape[-2:]:
            f_A2 = F.interpolate(f_A2, size=f_A1.shape[-2:], mode='bilinear', align_corners=False)
            f_B2 = F.interpolate(f_B2, size=f_A1.shape[-2:], mode='bilinear', align_corners=False)
        if f_A3.shape[-2:] != f_A1.shape[-2:]:
            f_A3 = F.interpolate(f_A3, size=f_A1.shape[-2:], mode='bilinear', align_corners=False)
            f_B3 = F.interpolate(f_B3, size=f_A1.shape[-2:], mode='bilinear', align_corners=False)
        if f_A4.shape[-2:] != f_A1.shape[-2:]:
            f_A4 = F.interpolate(f_A4, size=f_A1.shape[-2:], mode='bilinear', align_corners=False)
            f_B4 = F.interpolate(f_B4, size=f_A1.shape[-2:], mode='bilinear', align_corners=False)
        f_A1 = self.conv1(f_A1)
        f_B1 = self.conv1(f_B1)
        f_A2 = self.conv2(f_A2)
        f_B2 = self.conv2(f_B2)
        f_A3 = self.conv3(f_A3)
        f_B3 = self.conv3(f_B3)
        f_A4 = self.conv4(f_A4)
        f_B4 = self.conv4(f_B4)
        if self.CFI:
            f_AB1 = self.CFI_1(f_A1, f_B1)
            f_AB2 = self.CFI_2(f_A2, f_B2)
            f_AB3 = self.CFI_3(f_A3, f_B3)
            f_AB4 = self.CFI_4(f_A4, f_B4)
        else:
            f_AB1 = self.CFI_1(torch.cat([f_A1, f_B1], dim=1))
            f_AB2 = self.CFI_2(torch.cat([f_A2, f_B2], dim=1))
            f_AB3 = self.CFI_3(torch.cat([f_A3, f_B3], dim=1))
            f_AB4 = self.CFI_4(torch.cat([f_A4, f_B4], dim=1))
        if self.MFDS:
            f_AB4 = self.MFDSM(f_AB1, f_AB2, f_AB3, f_AB4)

        result = self.classifier(f_AB4)
        result_pred = F.interpolate(result, size=input_shape, mode='bilinear', align_corners=False)

        if self.CMR:
            result_pred = self.conv_out(self.CMRM(result_pred))

        return [f_A1, f_A2, f_A3, f_A4], [f_B1, f_B2, f_B3, f_B4], result_pred
        # return result_pred


def build_segmentor(out_layers=4, dilation=None, **kwargs):
    backbone_name = 'resnet50'
    pretrained_backbone = True
    backbone = None
    if backbone_name is not None:
        backbone = resnet.__dict__[backbone_name](
            input_channels=3,
            pretrained=pretrained_backbone,
            replace_stride_with_dilation=dilation)
        return_layers = {}
        for i in range(out_layers):
            return_layers['layer{}'.format(4 - i)] = 'layer{}'.format(4 - i)

        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = ChangeDetectionModel(backbone=backbone, dilation=dilation, **kwargs)
    return model


def Net(restore_from=None,
        map_location='cuda:0', **kwargs):
    model = build_segmentor(**kwargs)
    trainedwithdistribution = True
    if restore_from is not None:
        checkpoint = torch.load(restore_from, map_location=map_location)
        if trainedwithdistribution:
            checkpoint = checkpoint['CD_Net']['model_state']
            new_checkpoint = {}  ## 新建一个字典来访模型的权值
            for k, value in checkpoint.items():
                if k.split('.')[0] == 'basenet':
                    key = k.split("basenet.")[-1]
                    new_checkpoint[key] = value
            model.load_state_dict(new_checkpoint)
        else:
            model.load_state_dict(checkpoint['CD_Net']['model_state'])
        print("load checkpoint from {}".format(restore_from))

    return model
