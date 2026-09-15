import {parseBin, type Point} from './engine.ts';

export type SequenceFrame = {
  file: File;
  name: string;
  stem: string;
  sequenceId: string;
  index: number;
  timestamp: number;
  timingAssumed: boolean;
  labelFile?: File;
};

export type LoadedFrame = SequenceFrame & {
  points: Point[];
  semanticLabels?: number[];
  semanticNames?: Record<number, string>;
  semanticSource?: 'rellis-ground-truth' | 'model-prediction' | 'none';
};

export type Sequence = {
  id: string;
  name: string;
  frames: SequenceFrame[];
  assumedHz: number;
  timingAssumed: boolean;
};

export const RELLIS_NAMES:Record<number,string>={0:'void',1:'dirt',3:'grass',4:'tree',5:'pole',6:'water',7:'sky',8:'vehicle',9:'object',10:'asphalt',12:'building',15:'log',17:'person',18:'fence',19:'bush',23:'concrete',27:'barrier',31:'puddle',33:'mud',34:'rubble'};
const MAX_POINTS=250000;
const numberKey=(name:string)=>{const m=name.match(/(\d+)(?=\.[^.]+$)/);return m?Number(m[1]):Number.MAX_SAFE_INTEGER;};
const stem=(name:string)=>name.replace(/\.[^.]+$/,'');
const pathParts=(file:File)=>((file as File & {webkitRelativePath?:string}).webkitRelativePath||file.name).split(/[\\/]/).filter(Boolean);
const sequenceIdFor=(file:File)=>{const parts=pathParts(file);const seq=parts.find(p=>/^\d{5}$/.test(p));if(seq)return seq;const k=parts.findIndex(p=>p.toLowerCase()==='os1_cloud_node_kitti_bin');return k>0?parts[k-1]:parts.length>1?parts[0]:'upload';};

export function groupSequences(files:File[], hz:number):Sequence[]{
  const bins=files.filter(f=>f.name.toLowerCase().endsWith('.bin'));
  const labels=new Map<string,File>();
  for(const file of files.filter(f=>f.name.toLowerCase().endsWith('.label'))){
    const sid=sequenceIdFor(file);
    labels.set(`${sid}/${stem(file.name)}`,file);
  }
  const groups=new Map<string,File[]>();
  for(const file of bins){const sid=sequenceIdFor(file);groups.set(sid,[...(groups.get(sid)||[]),file]);}
  return [...groups.entries()].sort(([a],[b])=>a.localeCompare(b,undefined,{numeric:true})).map(([id,items])=>{
    const ordered=items.sort((a,b)=>numberKey(a.name)-numberKey(b.name)||a.name.localeCompare(b.name));
    const frames=ordered.map((file,index)=>({file,name:file.name,stem:stem(file.name),sequenceId:id,index,timestamp:index/hz,timingAssumed:true,labelFile:labels.get(`${id}/${stem(file.name)}`)}));
    return {id,name:id==='upload'?'Uploaded sequence':`RELLIS sequence ${id}`,frames,assumedHz:hz,timingAssumed:true};
  });
}

export function sequenceFromSingle(file:File,hz:number):Sequence{
  return {id:`single:${file.name}`,name:file.name,assumedHz:hz,timingAssumed:true,frames:[{file,name:file.name,stem:stem(file.name),sequenceId:`single:${file.name}`,index:0,timestamp:0,timingAssumed:true}]};
}

export async function parseRellisLabel(file:File,count:number){
  const buffer=await file.arrayBuffer();
  if(buffer.byteLength!==count*4)throw Error(`RELLIS label count mismatch for ${file.name}. Expected ${count}, got ${buffer.byteLength/4}.`);
  const raw=new Uint32Array(buffer);
  const semantic:number[]=[];
  for(const v of raw)semantic.push(v&0xffff);
  return semantic;
}

export class FrameCache{
  private cache=new Map<string,Promise<LoadedFrame>>();
  private max:number;
  constructor(max=5){this.max=max;}
  private key(frame:SequenceFrame){return `${frame.sequenceId}/${frame.name}/${frame.file.size}/${frame.file.lastModified}`;}
  load(frame:SequenceFrame):Promise<LoadedFrame>{
    const key=this.key(frame);
    let value=this.cache.get(key);
    if(!value){
      value=this.read(frame);
      this.cache.set(key,value);
      while(this.cache.size>this.max)this.cache.delete(this.cache.keys().next().value as string);
    }
    return value;
  }
  prefetch(frames:SequenceFrame[],center:number){
    for(const i of [center+1,center+2,center-1])if(frames[i])void this.load(frames[i]).catch(()=>{});
  }
  clear(){this.cache.clear();}
  private async read(frame:SequenceFrame):Promise<LoadedFrame>{
    const points=parseBin(await frame.file.arrayBuffer());
    if(points.length>MAX_POINTS)throw Error(`Frame ${frame.name} exceeds ${MAX_POINTS.toLocaleString()} points.`);
    let semanticLabels:number[]|undefined;
    if(frame.labelFile)semanticLabels=await parseRellisLabel(frame.labelFile,points.length);
    return {...frame,points,semanticLabels,semanticNames:semanticLabels?RELLIS_NAMES:undefined,semanticSource:semanticLabels?'rellis-ground-truth':'none'};
  }
}

export function timestampFor(frame:SequenceFrame,hz:number){return frame.timingAssumed?frame.index/hz:frame.timestamp;}
