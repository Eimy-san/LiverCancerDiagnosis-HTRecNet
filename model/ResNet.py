from torch import nn
import torch



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

    def __init__(self,inchannel,outchannel,stride=1,downsample=None):
        super(BottleNeck, self).__init__()

        self.conv1 = nn.Conv2d(inchannel,outchannel,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.conv2 = nn.Conv2d(outchannel,outchannel,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(outchannel)
        self.conv3 = nn.Conv2d(outchannel,outchannel*self.expansion,kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(outchannel*self.expansion)

        self.downsample = downsample
        self.relu = nn.ReLU(True)

    def forward(self,x):
        identity = x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.downsample is not None:
            identity = self.downsample(identity)

        output = self.relu(x+identity)
        return output



class ResNet(nn.Module):
    def __init__(self,residual,num_residuals,num_classes=10,include_top=True):
        super(ResNet, self).__init__()

        self.out = 64
        self.top = include_top

        self.conv1 = nn.Conv2d(3,self.out,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.out)
        self.relu = nn.ReLU(True)
        self.maxpooling = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.conv2 = self.residual_block(residual,64,num_residuals[0])
        self.conv3 = self.residual_block(residual,128,num_residuals[1],stride=2)
        self.conv4 = self.residual_block(residual,256,num_residuals[2],stride=2)
        self.conv5 = self.residual_block(residual,512,num_residuals[3],stride=2)

        if self.top:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(512*residual.expansion,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def residual_block(self,residual,channel,num_residuals,stride=1):
        downsample = None

        if stride != 1 or self.out != residual.expansion * channel:
            downsample = nn.Sequential(
                nn.Conv2d(self.out,channel * residual.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(channel*residual.expansion)
            )

        block = []

        block.append(residual(self.out,channel,stride=stride,downsample=downsample))
        self.out = channel*residual.expansion

        for _ in range(1,num_residuals):
            block.append(residual(self.out,channel))

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


def resnet18(num_classes=2,include_top=True):
    return ResNet(BasicBlock,[2,2,2,2],num_classes=num_classes,include_top=include_top)

def resnet34(num_classes=2,include_top=True):
    return ResNet(BasicBlock,[3,4,6,3],num_classes=num_classes,include_top=include_top)

def resnet50(num_classes=2, include_top=True):
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

