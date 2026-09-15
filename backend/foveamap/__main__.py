import argparse,json
from pathlib import Path
from time import perf_counter
import numpy as np
from .grid import build_grid,geometry,evaluate
from .io import load_scan,load_semantickitti,write_csv
from .simulation import generate
from .tracking import clusters,Tracker

def main():
    parser=argparse.ArgumentParser(description='FoveaMap LiDAR mapping pipeline')
    sub=parser.add_subparsers(dest='command',required=True)
    demo=sub.add_parser('demo');demo.add_argument('--output',default='demo');demo.add_argument('--frames',type=int,default=3)
    process=sub.add_parser('process');process.add_argument('input');process.add_argument('--output',default='grid.json');process.add_argument('--labels');process.add_argument('--checkpoint');process.add_argument('--demo-mlp',action='store_true');process.add_argument('--device',default='cpu');process.add_argument('--ground-offset',type=float,default=0);process.add_argument('--use-annotations',action='store_true')
    sequence=sub.add_parser('sequence');sequence.add_argument('input');sequence.add_argument('--output',default='sequence-output');sequence.add_argument('--hz',type=float,default=10);sequence.add_argument('--use-annotations',action='store_true')
    export=sub.add_parser('export-onnx');export.add_argument('checkpoint');export.add_argument('--output',default='foveamap.onnx')
    a=parser.parse_args()
    if a.command=='demo':
        if not 1<=a.frames<=300:parser.error('Choose 1..300 frames')
        out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
        for i in range(a.frames):
            p,y=generate(i);write_csv(out/f'{i:06}.csv',p,y)
            grid=build_grid(p,y);grid['evaluation']=evaluate(y,geometry(p),p);grid['semantic_source']='synthetic ground truth';(out/f'{i:06}.json').write_text(json.dumps(grid,indent=2))
        print(f'Generated {a.frames} annotated scans and maps in {out}')
    elif a.command=='process':
        start=perf_counter();p,truth=load_scan(a.input);p[:,2]-=a.ground_offset
        if a.labels:truth=load_semantickitti(a.labels,len(p))
        inference={}
        if a.checkpoint:
            from .model import infer
            pred,inference=infer(p,a.checkpoint,a.device);source='range CNN checkpoint'
        elif a.demo_mlp:
            from .point_mlp import predict
            pred=predict(p);source='synthetic-trained point MLP'
        elif a.use_annotations:
            if truth is None:parser.error('--use-annotations requires labels')
            pred=np.where(truth>=0,truth,geometry(p));source='input annotations, geometry fallback for ignored IDs'
        else:pred=geometry(p);source='geometric baseline'
        grid=build_grid(p,pred);grid['semantic_source']=source;grid['inference']=inference
        if truth is not None:grid['evaluation']=evaluate(truth,pred,p)
        grid['objects']=clusters(p,pred);grid['pipeline_ms']=(perf_counter()-start)*1000
        Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(grid,indent=2))
        print(json.dumps(grid['metrics'],indent=2))
    elif a.command=='sequence':
        if a.hz<=0:parser.error('--hz must be positive')
        paths=sorted(Path(a.input).glob('*.csv'))+sorted(Path(a.input).glob('*.bin'))
        if not paths:parser.error('No input scans')
        out=Path(a.output);out.mkdir(parents=True,exist_ok=True);tracker=Tracker()
        for i,path in enumerate(paths):
            p,y=load_scan(path)
            if a.use_annotations and y is None:parser.error(f'No annotations in {path}')
            pred=y if a.use_annotations else geometry(p)
            grid=build_grid(p,pred);grid['tracks']=tracker.update(clusters(p,pred),i/a.hz)
            grid['tracking_assumption']='stationary sensor / common coordinate frame; ego motion must be compensated before use'
            (out/f'{i:06}.json').write_text(json.dumps(grid))
        print(f'Processed {len(paths)} frames')
    else:
        import torch
        from .model import load_model
        model,projection=load_model(a.checkpoint)
        example=torch.zeros(1,5,projection['height'],projection['width'])
        torch.onnx.export(model,example,a.output,input_names=['range_image'],output_names=['class_logits'],opset_version=17,dynamo=False)
        Path(a.output+'.json').write_text(json.dumps({'projection':projection,'normalisation':'range/100, x/100, y/100, z/10, intensity','classes':['drivable','non-drivable','static','dynamic']}))
        print(f'Exported {a.output}; validate numerical parity on the target runtime before optimisation')
if __name__=='__main__':main()
