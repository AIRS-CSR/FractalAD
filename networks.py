
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, resnet34, resnet50, wide_resnet50_2, resnext50_32x4d
from torchvision.models import swin_t, swin_s, swin_b
from torchvision.models import convnext_tiny, convnext_small, convnext_base
from torchvision.models import efficientnet_v2_s
from torchvision.models import mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
from torchvision.models import shufflenet_v2_x0_5,shufflenet_v2_x1_0,shufflenet_v2_x1_5,shufflenet_v2_x2_0
from torchvision.models import vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16,vgg16_bn
 
#imagenet
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

class BackBone(nn.Module):

    def __init__(self, backbone='resnet18', pretrained=True):
        super(BackBone, self).__init__()

        if backbone.__contains__('resnet'):
            if backbone == 'resnet18':
                model = resnet18(weights='DEFAULT') if pretrained else resnet18()
                chs = [64, 128, 256, 512]
            if backbone == 'resnet34':
                model = resnet34(weights='DEFAULT') if pretrained else resnet34()
                chs = [64, 128, 256, 512]
            if backbone == 'resnet50':
                model = resnet50(weights='DEFAULT') if pretrained else resnet50()
                chs = [256, 512, 1024, 2048]
            if backbone == 'wide_resnet50_2':
                model = wide_resnet50_2(weights='DEFAULT') if pretrained else wide_resnet50_2()
                chs = [256, 512, 1024, 2048]
            if backbone == 'resnext50_32x4d':
                model = resnext50_32x4d(weights='DEFAULT') if pretrained else resnext50_32x4d()
                chs = [256, 512, 1024, 2048]
            layers = list(model.children())
            # print(layers)
            self.stage1 = torch.nn.Sequential(*layers[0:5])
            self.stage2 = torch.nn.Sequential(*layers[5:6])
            self.stage3 = torch.nn.Sequential(*layers[6:7])
            self.stage4 = torch.nn.Sequential(*layers[7:8])
        elif backbone.__contains__('convnext'):
            chs = [96, 192, 384, 768]
            if backbone == 'convnext_tiny':
                model = convnext_tiny(weights='DEFAULT') if pretrained else convnext_tiny()
            if backbone == 'convnext_small':
                model = convnext_small(weights='DEFAULT') if pretrained else convnext_small()
            if backbone == 'convnext_base':
                model = convnext_base(weights='DEFAULT') if pretrained else convnext_base()
            layers = list(model.children())
            # print(layers)
            self.stage1 = torch.nn.Sequential(*layers[0][0:2])
            self.stage2 = torch.nn.Sequential(*layers[0][2:4])
            self.stage3 = torch.nn.Sequential(*layers[0][4:6])
            self.stage4 = torch.nn.Sequential(*layers[0][6:8])
        elif backbone.__contains__('efficientnet'):
            if backbone=='efficientnet_v2_s':
                chs = [48, 64, 160, 1280]
                model = efficientnet_v2_s(weights='DEFAULT') if pretrained else efficientnet_v2_s()
                layers = list(model.children())
                self.stage1 = torch.nn.Sequential(*layers[0][0:3])
                self.stage2 = torch.nn.Sequential(*layers[0][3:4])
                self.stage3 = torch.nn.Sequential(*layers[0][4:6])
                self.stage4 = torch.nn.Sequential(*layers[0][6:8])

    def forward(self, x):

        x = transforms.Normalize(mean=mean_train, std=std_train)(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        return x1, x2, x3, x4


class SFNet(nn.Module):

    def __init__(self, backbone='resnet18', num_classes=1):
        super(SFNet, self).__init__()

        if backbone.__contains__('resnet'):
            if backbone == 'resnet18' or backbone == 'resnet34':
                chs = [512, 256, 128, 64]
            elif backbone == 'resnet50' or backbone == 'wide_resnet50_2' or backbone == 'resnext50_32x4d':
                chs = [2048, 1024, 512, 256]
        elif backbone.__contains__('convnext'):
            chs = [768, 384, 192, 96]
        elif backbone.__contains__('efficientnet'):
            if backbone == 'efficientnet_v2_s':
                chs = [1280, 160, 64, 48]

        # Top layer
        self.toplayer = nn.Conv2d(chs[0], chs[3], kernel_size=1, stride=1, padding=0)  # Reduce chs
        self.toplayer_bn = nn.BatchNorm2d(chs[3])

        # Lateral layers
        self.latlayer1 = nn.Conv2d(chs[1], chs[3], kernel_size=1, stride=1, padding=0)
        self.latlayer1_bn = nn.BatchNorm2d(chs[3])

        self.latlayer2 = nn.Conv2d(chs[2], chs[3], kernel_size=1, stride=1, padding=0)
        self.latlayer2_bn = nn.BatchNorm2d(chs[3])

        self.latlayer3 = nn.Conv2d(chs[3], chs[3], kernel_size=1, stride=1, padding=0)
        self.latlayer3_bn = nn.BatchNorm2d(chs[3])

        # Smooth layers
        self.smooth1 = nn.Conv2d(chs[3], chs[3], kernel_size=3, stride=1, padding=1)
        self.smooth1_bn = nn.BatchNorm2d(chs[3])

        self.smooth2 = nn.Conv2d(chs[3], chs[3], kernel_size=3, stride=1, padding=1)
        self.smooth2_bn = nn.BatchNorm2d(chs[3])

        self.smooth3 = nn.Conv2d(chs[3], chs[3], kernel_size=3, stride=1, padding=1)
        self.smooth3_bn = nn.BatchNorm2d(chs[3])
        
        # flow layer
        self.flowconv1 = nn.Conv2d(chs[3]*2, 2, kernel_size=3, stride=1, padding=1)
        self.flowconv2 = nn.Conv2d(chs[3]*2, 2, kernel_size=3, stride=1, padding=1)
        self.flowconv3 = nn.Conv2d(chs[3]*2, 2, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(4*chs[3], chs[3], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(chs[3])
        self.conv3 = nn.Conv2d(chs[3], num_classes, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        
    def upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(int(H // scale), int(W // scale)), mode='bilinear')

    def upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def coord_concat(self, x):
        x_range = torch.linspace(-1, 1, x.shape[-1], device=x.device)
        y_range = torch.linspace(-1, 1, x.shape[-2], device=x.device)
        y_range, x_range = torch.meshgrid(y_range, x_range)
        c1 = x_range.expand([x.shape[0], 1, -1, -1])
        c2 = y_range.expand([x.shape[0], 1, -1, -1])
        return torch.concat([x, c1, c2], dim=1)
    
    # Semantic Flow for Fast and Accurate Scene Parsing arXiv:2002.10120v1
    # Flow Align Module
    def flow_align_module(self, featmap_front, featmap_latter, func):
        B, C, H, W = featmap_latter.size()
        fuse = torch.cat((featmap_front, self.upsample(featmap_latter, featmap_front)), 1)
        
        flow = func(fuse)
        flow = self.upsample(flow, featmap_latter)
        flow = flow.permute(0, 2, 3, 1)
        grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        grid = torch.stack((grid_x, grid_y), 2).float()
        grid.requires_grad = False
        grid = grid.type_as(featmap_latter)
        vgrid = grid + flow
        # scale grid to [-1, 1]
        vgrid_x = 2.0 * vgrid[:,:,:,0] / max(W-1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:,:,:,1] / max(H-1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = F.grid_sample(featmap_latter, vgrid_scaled, mode='bilinear', padding_mode='zeros')
        return output

    def forward(self, x):
        
        c2, c3, c4, c5 = x

        # Top-down
        p5 = self.toplayer(c5)
        p5 = self.relu(self.toplayer_bn(p5))

        c4 = self.latlayer1(c4)
        c4 = self.relu(self.latlayer1_bn(c4))
        p5_flow = self.flow_align_module(c4, p5, self.flowconv1)
        p4 = self.upsample_add(p5_flow, c4)
        p4 = self.smooth1(p4)
        p4 = self.relu(self.smooth1_bn(p4))

        c3 = self.latlayer2(c3)
        c3 = self.relu(self.latlayer2_bn(c3))
        p4_flow = self.flow_align_module(c3, p4, self.flowconv2)
        p3 = self.upsample_add(p4_flow, c3)
        p3 = self.smooth2(p3)
        p3 = self.relu(self.smooth2_bn(p3))

        c2 = self.latlayer3(c2)
        c2 = self.relu(self.latlayer3_bn(c2))
        p3_flow = self.flow_align_module(c2, p3, self.flowconv3)
        p2 = self.upsample_add(p3_flow, c2)
        p2 = self.smooth3(p2)
        p2 = self.relu(self.smooth3_bn(p2))

        p3 = self.upsample(p3, p2)
        p4 = self.upsample(p4, p2)
        p5 = self.upsample(p5, p2)

        out = torch.cat((p2, p3, p4, p5), 1)
        out = self.conv2(out)
        out = self.relu(self.bn2(out))
        out = self.conv3(out)
        out = self.upsample(out, out, scale=0.25)


        return out