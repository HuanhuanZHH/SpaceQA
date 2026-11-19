"""All-family multi-label students and explicitly adapted spatial backbones."""
import torch
from torch import nn
from torch.nn import functional as F

from common import FEATURES,ROOT,REPO,module


class Student(nn.Module):
    def __init__(self,widths=(32,64,128),hidden=128):
        super().__init__();self.widths=list(widths);self.hidden=hidden
        self.convs=nn.ModuleList([nn.Conv2d(a,b,3,padding=1) for a,b in zip([3]+list(widths[:-1]),widths)])
        self.fc1=nn.Linear(widths[-1]*32+FEATURES,hidden);self.fc2=nn.Linear(hidden,3)
    def forward(self,x,features):
        for conv in self.convs:x=F.max_pool2d(F.gelu(conv(x)),2)
        x=F.adaptive_avg_pool2d(x,(4,8)).flatten(1)
        return self.fc2(F.dropout(F.gelu(self.fc1(torch.cat([x,features],1))),.1,self.training)),None


def prune_student(model):
    widths=[max(4,w//2) for w in model.widths];slim=Student(widths,model.hidden//2).to(next(model.parameters()).device)
    with torch.no_grad():
        previous=torch.arange(3,device=next(model.parameters()).device)
        for old,new in zip(model.convs,slim.convs):
            keep=old.weight.abs().sum((1,2,3)).topk(new.out_channels).indices.sort().values
            new.weight.copy_(old.weight[keep][:,previous]);new.bias.copy_(old.bias[keep]);previous=keep
        columns=(previous[:,None]*32+torch.arange(32,device=previous.device)[None]).flatten()
        columns=torch.cat([columns,torch.arange(model.widths[-1]*32,model.widths[-1]*32+FEATURES,device=previous.device)])
        keep=model.fc1.weight.abs().sum(1).topk(slim.hidden).indices.sort().values
        slim.fc1.weight.copy_(model.fc1.weight[keep][:,columns]);slim.fc1.bias.copy_(model.fc1.bias[keep])
        slim.fc2.weight.copy_(model.fc2.weight[:,keep]);slim.fc2.bias.copy_(model.fc2.bias)
    return slim


class Spatial(nn.Module):
    def __init__(self,name,pretrained=True):
        super().__init__();self.name=name;self.activation=None
        source=module('allfamily_spatial_models',REPO/'experiments/image_models_20261031/models.py')
        if name=='unet':
            class PlainUNet(source.AttentionUNet):
                def forward(self,x):
                    a=self.e1(x);b=self.e2(F.max_pool2d(a,2));c=self.e3(F.max_pool2d(b,2));d=self.e4(F.max_pool2d(c,2))
                    d=self.d3(torch.cat([source.up(d,c),c],1));d=self.d2(torch.cat([source.up(d,b),b],1))
                    return self.out(self.d1(torch.cat([source.up(d,a),a],1)))
            self.net=PlainUNet()
            # Remove unused attention modules physically rather than count dead parameters.
            del self.net.g1,self.net.g2,self.net.g3
        else:self.net=source.build(name,pretrained=pretrained)
        last=self.net.net.classifier[-1] if name=='deeplab_mobilenet' else self.net.out
        self.channels=last.in_channels
        last.register_forward_pre_hook(self.capture)
        self.head=nn.Sequential(nn.Linear(self.channels*4+FEATURES,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,3))
    def capture(self,module,args):self.activation=args[0]
    def forward(self,x,features):
        mask=self.net(x)
        z=F.adaptive_avg_pool2d(self.activation,(2,2)).flatten(1)
        return self.head(torch.cat([z,features],1)),mask


class YOLOBackbone(nn.Module):
    """Keep native multiscale backbone initialization, replace task heads explicitly."""
    def __init__(self,name,pretrained=True):
        super().__init__()
        from ultralytics import YOLO
        weights={'yolo11_detect_backbone':'yolo11n.pt','yolo11_segment_backbone':'yolo11n-seg.pt','yolov8_segment_backbone':'yolov8n-seg.pt'}
        wrapper=YOLO(str(ROOT/'models/image-model-pilot'/weights[name]))
        net=wrapper.model
        cutoff=int(net.yaml['backbone'].__len__())
        self.layers=nn.ModuleList(list(net.model.children())[:cutoff])
        for p in self.layers.parameters():p.requires_grad_(True)
        self.head=nn.Sequential(nn.Linear(256*4+FEATURES,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,3))
        # Infer final channel count using real backbone routing, never assume a YOLO version layout.
        self.layers.eval()
        with torch.no_grad():c=self.encode(torch.zeros(2,3,128,256)).shape[1]
        self.head[0]=nn.Linear(c*4+FEATURES,128)
        self.mask=nn.Sequential(nn.Conv2d(c,32,1),nn.SiLU(),nn.Conv2d(32,1,1))
    def encode(self,x):
        outputs=[]
        for layer in self.layers:
            if layer.f!=-1:x=outputs[layer.f] if isinstance(layer.f,int) else [x if j==-1 else outputs[j] for j in layer.f]
            x=layer(x);outputs.append(x)
        return x
    def forward(self,x,features):
        z=self.encode(x)
        mask=F.interpolate(self.mask(z),size=x.shape[-2:],mode='bilinear',align_corners=False)
        return self.head(torch.cat([F.adaptive_avg_pool2d(z,(2,2)).flatten(1),features],1)),mask


class TransformerSpatial(nn.Module):
    def __init__(self,name,pretrained=True):
        super().__init__()
        from transformers import Dinov2Model,IJepaModel
        cls=Dinov2Model if name=='dino_finetune' else IJepaModel
        folder='dinov2-small' if name=='dino_finetune' else 'ijepa_vith14_1k'
        self.backbone=cls.from_pretrained(ROOT/'models'/folder,local_files_only=True,torch_dtype=torch.bfloat16)
        for p in self.backbone.parameters():p.requires_grad_(False)
        for block in self.backbone.encoder.layer[-2:]:
            for p in block.parameters():p.requires_grad_(True)
        self.has_class_token=name=='dino_finetune';channels=self.backbone.config.hidden_size
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
        self.head=nn.Sequential(nn.Linear(channels+FEATURES,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,3))
        self.mask=nn.Sequential(nn.Conv2d(channels,64,1),nn.SiLU(),nn.Conv2d(64,1,1))
    def forward(self,x,features):
        pixels=F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False)
        z=self.backbone(pixel_values=(pixels-self.mean)/self.std).last_hidden_state
        if self.has_class_token:z=z[:,1:]
        width=int(z.shape[1]**.5);assert width*width==z.shape[1]
        dense=z.transpose(1,2).reshape(len(x),z.shape[-1],width,width)
        logits=self.head(torch.cat([z.mean(1),features],1))
        mask=F.interpolate(self.mask(dense),size=x.shape[-2:],mode='bilinear',align_corners=False)
        return logits,mask


def build(route,pretrained=True,widths=None,hidden=None):
    if route.split('_seed')[0] in ['supervised','distilled','pruned']:
        return Student(widths or ((16,32,64) if route.startswith('pruned') else (32,64,128)),hidden or (64 if route.startswith('pruned') else 128))
    if route.startswith('yolo'):return YOLOBackbone(route,pretrained)
    if route in ['dino_finetune','ijepa_finetune']:return TransformerSpatial(route,pretrained)
    return Spatial(route,pretrained)
