import torch
import numpy as np
from .model import RangeNet
from .range_image import range_image, range_image_inverse  # see note below

_model = None
_projection = None

def load_checkpoint(path, device='cpu'):
    global _model, _projection
    ckpt = torch.load(path, map_location=device)
    model = RangeNet()
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    _model = model
    _projection = ckpt['projection']
    return ckpt

def infer(points, device='cpu'):
    if _model is None:
        raise RuntimeError('No checkpoint loaded')
    x, _, index_map = range_image(points, None, **_projection)  # see open question below
    with torch.no_grad():
        logits = _model(torch.from_numpy(x).unsqueeze(0).to(device))
        pred_grid = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # [H, W]
    # unproject grid predictions back to per-point labels using index_map
    semantic = pred_grid.flatten()[index_map]  # per-point class, -1 where projection missed
    return semantic