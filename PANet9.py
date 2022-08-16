import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
"""
CAA-Net
the code is modified from ResNet
"""


__all__ = ['ResNet', 'resnet34']



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.cls_layer = []
        self.fc_layer = []
        self.x_cls = []
        self.x_fc = []
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.cls_layer1 = self._make_layer(block, 256, layers[3], stride=1)
        self.cls_layer2 = self._make_layer(block, 256, layers[3], stride=1)
        self.cls_layer3 = self._make_layer(block, 256, layers[3], stride=1)
        self.cls_layer4 = self._make_layer(block, 256, layers[3], stride=1)

        if block == Bottleneck:
            self.fc_layer11 = nn.Linear(1024, 64)
            self.fc_layer21 = nn.Linear(1024, 64)
            self.fc_layer31 = nn.Linear(1024, 64)
            self.fc_layer41 = nn.Linear(1024, 64)
            ##
            self.fc_layer12 = nn.Linear(64, 1)
            self.fc_layer22 = nn.Linear(64, 1)
            self.fc_layer32 = nn.Linear(64, 1)
            self.fc_layer42 = nn.Linear(64, 1)
            self.inplanes = 5120
            self.fc_cat=4096
        else:
            self.fc_layer11 = nn.Linear(256, 64)
            self.fc_layer21 = nn.Linear(256, 64)
            self.fc_layer31 = nn.Linear(256, 64)
            self.fc_layer41 = nn.Linear(256, 64)
            ##
            self.fc_layer12 = nn.Linear(64, 1)
            self.fc_layer22 = nn.Linear(64, 1)
            self.fc_layer32 = nn.Linear(64, 1)
            self.fc_layer42 = nn.Linear(64, 1)
            self.inplanes = 1280
            self.fc_cat=1024
        #####
        self.softmax = nn.Softmax(dim=1)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_1 = nn.Linear(512 * block.expansion+self.fc_cat, 224)
        self.fc_2 = nn.Linear(224, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        ###########
        x_cls1 = self.cls_layer1(x)
        xfc = self.avg_pool(x_cls1)
        xfc1 = xfc.view(xfc.size(0), -1)
        x_fc1 = self.fc_layer11(xfc1)
        x_fc1 = self.fc_layer12(x_fc1)
        ####
        x_cls2 = self.cls_layer2(x)
        xfc = self.avg_pool(x_cls2)
        xfc2 = xfc.view(xfc.size(0), -1)
        x_fc2 = self.fc_layer21(xfc2)
        x_fc2 = self.fc_layer22(x_fc2)
        ####
        x_cls3 = self.cls_layer3(x)
        xfc = self.avg_pool(x_cls3)
        xfc3 = xfc.view(xfc.size(0), -1)
        x_fc3 = self.fc_layer31(xfc3)
        x_fc3 = self.fc_layer32( x_fc3)
        ####
        x_cls4 = self.cls_layer4(x)
        xfc = self.avg_pool(x_cls4)
        xfc4 = xfc.view(xfc.size(0), -1)
        x_fc4 = self.fc_layer41(xfc4)
        x_fc4 = self.fc_layer42(x_fc4)
        ####
        x_fc=torch.cat((x_fc1, x_fc2, x_fc3, x_fc4), dim=1)
        x_fc = self.softmax(x_fc)
        ####
        x_cls1 = torch.exp(x_cls1)
        x_cls1 = x_cls1.div(torch.sum(x_cls1, dim=(2, 3), keepdim=True))
        x_cls1 = x * x_cls1
        w1=x_fc[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        x_cls1=w1 * x_cls1
        ####
        x_cls2 = torch.exp(x_cls2)
        x_cls2 = x_cls2.div(torch.sum(x_cls2, dim=(2, 3), keepdim=True))
        x_cls2 = x * x_cls2
        w2 = x_fc[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        x_cls2 = w2 * x_cls2
        ####
        x_cls3 = torch.exp(x_cls3)
        x_cls3 = x_cls3.div(torch.sum(x_cls3, dim=(2, 3), keepdim=True))
        x_cls3 = x * x_cls3
        w3 = x_fc[:, 2].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        x_cls3 = w3 * x_cls3
        ####
        x_cls4 = torch.exp(x_cls4)
        x_cls4 = x_cls4.div(torch.sum(x_cls4, dim=(2, 3), keepdim=True))
        x_cls4 = x * x_cls4
        w4 = x_fc[:, 3].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        x_cls4 = w4 * x_cls4
################

        x = torch.cat((x, x_cls1, x_cls2, x_cls3, x_cls4), dim=1)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.cat((x,xfc1,xfc2,xfc3,xfc4),dim=1)
        x = self.fc_1(x)
        x = self.fc_2(x)


        return x,x_fc

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    # if pretrained:
    return model




def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""
    its real name is CAA-Net
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


