import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from .model import RangeNet
from .range_image import range_image
from .io import load_scan,load_semantickitti

class Scans(Dataset):
    def __init__(self,root,projection):
        self.paths=sorted(Path(root).glob('*.csv'))+sorted(Path(root).glob('*.bin'))
        if not self.paths: raise ValueError(f'No CSV/BIN scans in {root}')
        self.projection=projection
    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        p,y=load_scan(self.paths[i])
        if y is None:
            path=self.paths[i].with_suffix('.label')
            if not path.exists(): raise ValueError(f'Ground-truth labels required: {path}')
            y=load_semantickitti(path,len(p))
        x,target,*_=range_image(p,y,**self.projection)
        if not (target>=0).any(): raise ValueError('Scan has no labels within configured field of view')
        return torch.from_numpy(x),torch.from_numpy(target)

def main():
    p=argparse.ArgumentParser(description='Train the four-class range-image CNN on labelled scans')
    p.add_argument('--train',required=True);p.add_argument('--val',required=True);p.add_argument('--output',default='checkpoints/best.pt');p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=2);p.add_argument('--device',default='cpu');p.add_argument('--width',type=int,default=512);p.add_argument('--height',type=int,default=64);p.add_argument('--fov-up',type=float,default=15);p.add_argument('--fov-down',type=float,default=-25);p.add_argument('--class-weights',type=float,nargs=4,default=[1,1,1,1])
    a=p.parse_args();torch.manual_seed(42);np.random.seed(42)
    if a.epochs<1 or a.batch_size<1 or a.width<4 or a.height<4: p.error('Epochs/batch size must be positive and dimensions >= 4')
    if min(a.class_weights)<=0: p.error('Class weights must be positive')
    projection=dict(height=a.height,width=a.width,fov_up=a.fov_up,fov_down=a.fov_down)
    train,val=Scans(a.train,projection),Scans(a.val,projection)
    if set(x.resolve() for x in train.paths)&set(x.resolve() for x in val.paths): raise ValueError('Training and validation files must be disjoint')
    model=RangeNet().to(a.device);optim=torch.optim.AdamW(model.parameters(),lr=1e-3)
    loss_fn=torch.nn.CrossEntropyLoss(ignore_index=-1,weight=torch.tensor(a.class_weights,dtype=torch.float32,device=a.device))
    best=float('inf');log=[];out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(a.epochs):
        losses={}
        for name,data in [('train',train),('val',val)]:
            model.train(name=='train');total=0
            for x,y in DataLoader(data,batch_size=a.batch_size,shuffle=name=='train'):
                x,y=x.to(a.device),y.to(a.device)
                with torch.set_grad_enabled(name=='train'):
                    loss=loss_fn(model(x),y)
                    if name=='train': optim.zero_grad();loss.backward();optim.step()
                total+=float(loss.detach())*len(x)
            losses[name]=total/len(data)
        record={'epoch':epoch+1,**losses};log.append(record);print(json.dumps(record),flush=True)
        if losses['val']<best:
            best=losses['val'];torch.save({'architecture':'foveamap-range-v1','state_dict':model.state_dict(),'projection':projection,'validation_loss':best,'epoch':epoch+1},out)
    out.with_suffix('.training.json').write_text(json.dumps(log,indent=2))
if __name__=='__main__':main()
