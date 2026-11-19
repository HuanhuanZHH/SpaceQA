"""Compact U-Net variants and explicitly initialized torchvision encoders."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

WEIGHTS=Path('/staff/tangjiarui/OPC/models/image-model-pilot')


class Conv(nn.Sequential):
    def __init__(self,a,b):
        super().__init__(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU(),
                         nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU())


def up(x,like):return F.interpolate(x,size=like.shape[-2:],mode='bilinear',align_corners=False)


class Gate(nn.Module):
    def __init__(self,skip,gate):
        super().__init__();mid=max(4,skip//2)
        self.s=nn.Conv2d(skip,mid,1);self.g=nn.Conv2d(gate,mid,1);self.score=nn.Conv2d(mid,1,1)
    def forward(self,s,g):return s*self.score(F.silu(self.s(s)+self.g(up(g,s)))).sigmoid()


class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__();self.e1=Conv(3,16);self.e2=Conv(16,32);self.e3=Conv(32,64);self.e4=Conv(64,128)
        self.g3=Gate(64,128);self.g2=Gate(32,64);self.g1=Gate(16,32)
        self.d3=Conv(192,64);self.d2=Conv(96,32);self.d1=Conv(48,16);self.out=nn.Conv2d(16,1,1)
    def forward(self,x):
        a=self.e1(x);b=self.e2(F.max_pool2d(a,2));c=self.e3(F.max_pool2d(b,2));d=self.e4(F.max_pool2d(c,2))
        d=self.d3(torch.cat([up(d,c),self.g3(c,d)],1))
        d=self.d2(torch.cat([up(d,b),self.g2(b,d)],1))
        return self.out(self.d1(torch.cat([up(d,a),self.g1(a,d)],1)))


class NestedUNet(nn.Module):
    def __init__(self):
        super().__init__();self.c00=Conv(3,16);self.c10=Conv(16,32);self.c20=Conv(32,64);self.c30=Conv(64,128)
        self.c01=Conv(48,16);self.c11=Conv(96,32);self.c21=Conv(192,64)
        self.c02=Conv(64,16);self.c12=Conv(128,32);self.c03=Conv(80,16);self.out=nn.Conv2d(16,1,1)
    def forward(self,x):
        a=self.c00(x);b=self.c10(F.max_pool2d(a,2));c=self.c20(F.max_pool2d(b,2));d=self.c30(F.max_pool2d(c,2))
        a1=self.c01(torch.cat([a,up(b,a)],1));b1=self.c11(torch.cat([b,up(c,b)],1));c1=self.c21(torch.cat([c,up(d,c)],1))
        a2=self.c02(torch.cat([a,a1,up(b1,a)],1));b2=self.c12(torch.cat([b,b1,up(c1,b)],1))
        return self.out(self.c03(torch.cat([a,a1,a2,up(b2,a)],1)))


class Normalize(nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
    def forward(self,x):return (x-self.mean)/self.std


class ResNetUNet(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__();self.norm=Normalize();e=models.resnet18(weights=None)
        if pretrained:e.load_state_dict(torch.load(WEIGHTS/'resnet18-f37072fd.pth',map_location='cpu',weights_only=True))
        self.stem=nn.Sequential(e.conv1,e.bn1,e.relu);self.pool=e.maxpool
        self.e1=e.layer1;self.e2=e.layer2;self.e3=e.layer3;self.e4=e.layer4
        self.d3=Conv(768,128);self.d2=Conv(256,64);self.d1=Conv(128,32);self.d0=Conv(96,16)
        self.raw=nn.Conv2d(3,16,3,padding=1);self.final=Conv(32,16);self.out=nn.Conv2d(16,1,1)
    def forward(self,x):
        a=self.stem(self.norm(x));b=self.e1(self.pool(a));c=self.e2(b);d=self.e3(c);e=self.e4(d)
        z=self.d3(torch.cat([up(e,d),d],1));z=self.d2(torch.cat([up(z,c),c],1));z=self.d1(torch.cat([up(z,b),b],1))
        z=self.d0(torch.cat([up(z,a),a],1));return self.out(self.final(torch.cat([up(z,x),self.raw(x)],1)))


class DeepLabMobileNet(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__();self.norm=Normalize()
        net=models.segmentation.deeplabv3_mobilenet_v3_large(weights=None,weights_backbone=None,num_classes=21,aux_loss=True)
        if pretrained:net.load_state_dict(torch.load(WEIGHTS/'deeplabv3_mobilenet_v3_large-fc3c493d.pth',map_location='cpu',weights_only=True))
        net.classifier[-1]=nn.Conv2d(256,1,1);net.aux_classifier=None;self.net=net
    def forward(self,x):return self.net(self.norm(x))['out']


def build(name,pretrained=True):
    return {'attention_unet':AttentionUNet,'unetpp':NestedUNet,
            'resnet18_unet':lambda:ResNetUNet(pretrained),'deeplab_mobilenet':lambda:DeepLabMobileNet(pretrained)}[name]()
