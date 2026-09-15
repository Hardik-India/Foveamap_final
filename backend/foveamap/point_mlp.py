"""Learned synthetic-demo classifier: two hidden layers, no neighbourhood or motion input."""
import json
from pathlib import Path
import numpy as np

def features(points):
    p=np.asarray(points,dtype=np.float32)
    return np.column_stack([p[:,0]/100,p[:,1]/100,p[:,2]/5,p[:,3],np.hypot(p[:,0],p[:,1])/100]).astype(np.float32)

def predict(points,checkpoint=None):
    path=Path(checkpoint) if checkpoint else Path(__file__).with_name('demo_weights.json')
    d=json.loads(path.read_text());x=features(points)
    for i in range(3):
        x=x@np.asarray(d[f'w{i}'],dtype=np.float32)+np.asarray(d[f'b{i}'],dtype=np.float32)
        if i<2:x=np.maximum(0,x)
    return x.argmax(1)

def train_demo(output,epochs=28):
    from .simulation import generate
    from .grid import evaluate
    rng=np.random.default_rng(2026)
    data=[generate(frame=i*13,seed=100+i,count=2000) for i in range(12)]
    p=np.vstack([a for a,b in data]);y=np.concatenate([b for a,b in data]);x=features(p)
    dims=[5,16,16,4];weights=[(rng.standard_normal((dims[i],dims[i+1]))*np.sqrt(2/dims[i])).astype(np.float32) for i in range(3)];bias=[np.zeros(d,dtype=np.float32) for d in dims[1:]]
    mw=[np.zeros_like(v) for v in weights];vw=[np.zeros_like(v) for v in weights];mb=[np.zeros_like(v) for v in bias];vb=[np.zeros_like(v) for v in bias];t=0
    class_weight=(len(y)/(4*np.bincount(y,minlength=4)))**.5
    for epoch in range(epochs):
        order=rng.permutation(len(y));loss_total=0
        for start in range(0,len(y),512):
            ids=order[start:start+512];a=[x[ids]];zs=[]
            for i in range(3):
                z=a[-1]@weights[i]+bias[i];zs.append(z);a.append(np.maximum(z,0) if i<2 else z)
            scores=a[-1]-a[-1].max(1,keepdims=True);probs=np.exp(scores);probs/=probs.sum(1,keepdims=True)
            loss_total+=float((-np.log(np.maximum(probs[np.arange(len(ids)),y[ids]],1e-8))*class_weight[y[ids]]).sum())
            grad=probs;grad[np.arange(len(ids)),y[ids]]-=1;grad*=class_weight[y[ids],None]/len(ids)
            dw=[None]*3;db=[None]*3
            for i in reversed(range(3)):
                dw[i]=a[i].T@grad;db[i]=grad.sum(0)
                if i:grad=(grad@weights[i].T)*(zs[i-1]>0)
            t+=1
            for i in range(3):
                for params,grads,m,v in [(weights,dw,mw,vw),(bias,db,mb,vb)]:
                    m[i]=.9*m[i]+.1*grads[i];v[i]=.999*v[i]+.001*grads[i]**2
                    params[i]-=.003*(m[i]/(1-.9**t))/(np.sqrt(v[i]/(1-.999**t))+1e-8)
        if epoch%7==0:print(f'Epoch {epoch+1}: weighted loss {loss_total/len(y):.4f}',flush=True)
    result={'architecture':'point-mlp-5-16-16-4','training_source':'procedural synthetic scenes only','seed':2026,'features':['x/100','y/100','z/5','intensity','xy_range/100']}
    for i in range(3):result[f'w{i}']=weights[i].tolist();result[f'b{i}']=bias[i].tolist()
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result))
    vp,vy=generate(frame=79,seed=999,count=5000);metrics=evaluate(vy,predict(vp,out),vp)
    out.with_suffix('.validation.json').write_text(json.dumps({'dataset':'held-out procedural seed 999, frame 79; not real-world validation','points':len(vp),'evaluation':metrics},indent=2))
    print(json.dumps({'synthetic_validation_miou':metrics['miou'],'checkpoint':str(out)}))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(Path(__file__).with_name('demo_weights.json')));p.add_argument('--epochs',type=int,default=28);a=p.parse_args();train_demo(a.output,a.epochs)
