from torch import nn
import torch
from model.cbam import *

class SPConv_3x3(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, ratio=0.5):
        super(SPConv_3x3, self).__init__()
        self.inplanes_3x3 = int(inplanes * ratio)
        self.inplanes_1x1 = inplanes - self.inplanes_3x3
        self.outplanes_3x3 = int(outplanes * ratio)
        self.outplanes_1x1 = outplanes - self.outplanes_3x3
        self.outplanes = outplanes
        self.stride = stride

        self.gwc = nn.Conv2d(self.inplanes_3x3, self.outplanes, kernel_size=3, stride=self.stride,
                             padding=1, groups=2, bias=False)
        self.pwc = nn.Conv2d(self.inplanes_3x3, self.outplanes, kernel_size=1, bias=False)

        self.conv1x1 = nn.Conv2d(self.inplanes_1x1, self.outplanes, kernel_size=1)
        self.avgpool_s2_1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avgpool_s2_3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avgpool_add_1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool_add_3 = nn.AdaptiveAvgPool2d(1)
        self.bn1 = nn.BatchNorm2d(self.outplanes)
        self.bn2 = nn.BatchNorm2d(self.outplanes)
        self.ratio = ratio
        self.groups = int(1 / self.ratio)

    def forward(self, x):
        b, c, _, _ = x.size()

        x_3x3 = x[:, :int(c * self.ratio), :, :]
        x_1x1 = x[:, int(c * self.ratio):, :, :]
        out_3x3_gwc = self.gwc(x_3x3)
        if self.stride == 2:
            x_3x3 = self.avgpool_s2_3(x_3x3)
        out_3x3_pwc = self.pwc(x_3x3)
        out_3x3 = out_3x3_gwc + out_3x3_pwc
        out_3x3 = self.bn1(out_3x3)
        out_3x3_ratio = self.avgpool_add_3(out_3x3).squeeze(dim=3).squeeze(dim=2)

        # use avgpool first to reduce information lost
        if self.stride == 2:
            x_1x1 = self.avgpool_s2_1(x_1x1)

        out_1x1 = self.conv1x1(x_1x1)
        out_1x1 = self.bn2(out_1x1)
        out_1x1_ratio = self.avgpool_add_1(out_1x1).squeeze(dim=3).squeeze(dim=2)

        out_31_ratio = torch.stack((out_3x3_ratio, out_1x1_ratio), 2)
        out_31_ratio = nn.Softmax(dim=2)(out_31_ratio)
        out = out_1x1 * (out_31_ratio[:, :, 1].view(b, self.outplanes, 1, 1).expand_as(out_1x1)) \
              + out_3x3 * (out_31_ratio[:, :, 0].view(b, self.outplanes, 1, 1).expand_as(out_3x3))

        return out


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self,inchannel,outchannel,stride=1,downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inchannel,outchannel,kernel_size=3,padding=1,stride=stride,bias=False)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.conv2 = nn.Conv2d(outchannel,outchannel,kernel_size=3,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(outchannel)

        self.downsample = downsample
        self.relu = nn.ReLU(True)


    def forward(self,x):

        identity = x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.downsample is not None:
            identity = self.downsample(identity)

        output = self.relu(identity+x)

        return output



class BottleNeck(nn.Module):

    expansion = 4

    def __init__(self,inchannel,outchannel,stride=1,downsample=None,use_cbam=True):
        super(BottleNeck, self).__init__()

        self.conv1 = nn.Conv2d(inchannel,outchannel,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.conv2 = SPConv_3x3(outchannel,outchannel,stride=stride)
        self.bn2 = nn.BatchNorm2d(outchannel)
        self.conv3 = nn.Conv2d(outchannel,outchannel*self.expansion,kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(outchannel*self.expansion)

        self.downsample = downsample
        self.relu = nn.ReLU(True)

        if use_cbam:
            self.cbam = CBAM(outchannel*self.expansion, 16)
        else:
            self.cbam = None

    def forward(self,x):
        identity = x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.downsample is not None:
            identity = self.downsample(identity)
        if not self.cbam is None:
            x = self.cbam(x)
        output = self.relu(x+identity)
        return output



class Net(nn.Module):
    def __init__(self,residual,num_residuals,num_classes=10,include_top=True):
        super(Net, self).__init__()

        self.out = 64
        self.top = include_top

        self.conv1 = nn.Conv2d(3,self.out,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.out)
        self.relu = nn.ReLU(True)
        self.maxpooling = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.conv2 = self.residual_block(residual,64,num_residuals[0],use_cbam=True)
        self.conv3 = self.residual_block(residual,128,num_residuals[1],stride=2,use_cbam=True)
        self.conv4 = self.residual_block(residual,256,num_residuals[2],stride=2,use_cbam=True)
        self.conv5 = self.residual_block(residual,512,num_residuals[3],stride=2,)

        if self.top:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(512*residual.expansion,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def residual_block(self,residual,channel,num_residuals,stride=1,use_cbam=False):
        downsample = None

        if stride != 1 or self.out != residual.expansion * channel:
            downsample = nn.Sequential(
                nn.Conv2d(self.out,channel * residual.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(channel*residual.expansion)
            )

        block = []

        block.append(residual(self.out,channel,stride=stride,downsample=downsample,use_cbam=use_cbam))
        self.out = channel*residual.expansion

        for _ in range(1,num_residuals):
            block.append(residual(self.out,channel,use_cbam=use_cbam))

        return nn.Sequential(*block)

    def forward(self,x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpooling(x)

        x = self.conv5(self.conv4(self.conv3(self.conv2(x))))
        if self.top:
            x = self.avgpool(x)
            x = torch.flatten(x,1)
            x = self.fc(x)

        return x

def HTRecNet(num_classes=2, include_top=True):
    return Net(BottleNeck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

