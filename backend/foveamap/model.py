"""Compact range-image encoder/decoder; inspired by range CNNs, not SalsaNext weights."""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class RangeNet(nn.Module):
    def __init__(self, classes=18):
        super().__init__()
        def block(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.GroupNorm(4,o),nn.SiLU(),nn.Conv2d(o,o,3,padding=1),nn.GroupNorm(4,o),nn.SiLU())
        self.enc1=block(5,24);self.enc2=block(24,48);self.bottleneck=block(48,96)
        self.dec2=block(144,48);self.dec1=block(72,24);self.head=nn.Conv2d(24,classes,1)
    def forward(self,x):
        a=self.enc1(x);b=self.enc2(F.avg_pool2d(a,2));c=self.bottleneck(F.avg_pool2d(b,2))
        d=self.dec2(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        e=self.dec1(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        return self.head(e)

def load_model(checkpoint, device='cpu'):
    data = torch.load(checkpoint, map_location=device, weights_only=True)
    if data.get('architecture') != 'foveamap-range-v1':
        raise ValueError('Unsupported checkpoint architecture')
    num_classes = data.get('num_classes', 18)
    model = RangeNet(classes=num_classes).to(device)
    model.load_state_dict(data['state_dict'])
    model.eval()
    return model, data['projection'], data.get('class_names')

def infer(points, checkpoint, device='cpu'):
    from .range_image import range_image
    model, projection, class_names = load_model(checkpoint, device)
    features, _, row, col, valid = range_image(points, **projection)
    with torch.inference_mode():
        labels = model(torch.from_numpy(features[None]).to(device)).argmax(1)[0].cpu().numpy()
    out = np.full(len(points), -1, dtype=np.int64)
    out[valid] = labels[row[valid], col[valid]]
    return out, {
        'neural_points': int(valid.sum()),
        'unlabeled_points': int((~valid).sum()),
        'class_names': class_names,
    }