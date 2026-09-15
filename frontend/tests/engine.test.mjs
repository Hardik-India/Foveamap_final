import test from 'node:test';
import assert from 'node:assert/strict';
import {address,project,simulate,parseCSV,parseBin,DEFAULT} from '../lib/fovea/engine.ts';
import {groupSequences,parseRellisLabel,FrameCache} from '../lib/fovea/sequence.ts';
import {LocalTracker,extractObjects,TRACKING_DEFAULTS} from '../lib/fovea/tracking.ts';
test('exclusive band endpoints, far endpoint, invalid rejection',()=>{for(const [r,tier]of [[0,0],[9.999999,0],[10,1],[29.999999,1],[30,2],[100,2]]){const a=address({x:r,y:0,z:0});assert.equal(a.tier,tier);assert.ok(a.r0<=r+1e-10&&a.r1>=r);}assert.equal(address({x:100.001,y:0,z:0}),null);assert.equal(address({x:NaN,y:0,z:0}),null);});
test('height aggregation and exact conservation across boundaries',()=>{const p=[{x:10,y:0,z:1,label:2},{x:10,y:0,z:3,label:2},{x:30,y:0,z:0,label:0},{x:100,y:0,z:0,label:0}];const r=project(p);assert.equal(r.cells.reduce((a,c)=>a+c.count,0),4);const c=r.cells.find(c=>c.count===2);assert.equal(c.min,1);assert.equal(c.max,3);assert.equal(c.mean,2);assert.equal(c.roughness,1);assert.equal(c.confidence,1);});
test('random simulation loses and duplicates no in-range points',()=>{const p=simulate(42),r=project(p);assert.equal(r.accepted,p.filter(p=>Math.hypot(p.x,p.y)<=100).length);assert.equal(r.cells.reduce((n,c)=>n+c.count,0),r.accepted);assert.equal(new Set(r.cells.map(c=>c.key)).size,r.cells.length);});
test('CSV validates malformed, missing and invalid semantic values',()=>{assert.equal(parseCSV('x,y,z,label\n1,2,3,2')[0].label,2);assert.throws(()=>parseCSV('x,y,z\n,2,3'));assert.throws(()=>parseCSV('x,y,z,label\n1,2,3,7'));assert.throws(()=>parseCSV('x,y,z'));assert.throws(()=>parseCSV('x,y,z\nNaN,2,3'));});
test('KITTI binary shape and float values',()=>{const a=new Float32Array([1,2,3,.5]);assert.equal(parseBin(a.buffer)[0].intensity,.5);assert.throws(()=>parseBin(new ArrayBuffer(4)));assert.throws(()=>parseBin(new Float32Array([NaN,1,2,3]).buffer));});
test('unlabelled data never emits fabricated accuracy',()=>{assert.equal(project([{x:0,y:0,z:0}]).miou,null);});
test('configuration rejects inverted and non-finite tiers',()=>{assert.throws(()=>project([],{...DEFAULT,near:40}));assert.throws(()=>project([],{...DEFAULT,nearSize:0}));});
test('neural inference never consumes ground-truth labels',()=>{const a=project([{x:3,y:1,z:0,intensity:.6,label:0}],DEFAULT,'neural');const b=project([{x:3,y:1,z:0,intensity:.6,label:3}],DEFAULT,'neural');assert.deepEqual(a.pointLabels,b.pointLabels);assert.equal(a.cells[0].label,b.cells[0].label);});
test('sequence upload sorts numerically and keeps sequences separate',()=>{
 const files=[
  new File([new ArrayBuffer(16)],'000010.bin',{lastModified:1}),
  new File([new ArrayBuffer(16)],'000002.bin',{lastModified:1}),
  new File([new ArrayBuffer(16)],'000001.bin',{lastModified:1}),
 ];
 Object.defineProperty(files[0],'webkitRelativePath',{value:'00000/os1_cloud_node_kitti_bin/000010.bin'});
 Object.defineProperty(files[1],'webkitRelativePath',{value:'00000/os1_cloud_node_kitti_bin/000002.bin'});
 Object.defineProperty(files[2],'webkitRelativePath',{value:'00001/os1_cloud_node_kitti_bin/000001.bin'});
 const seqs=groupSequences(files,10);
 assert.equal(seqs.length,2);
 assert.deepEqual(seqs[0].frames.map(f=>f.name),['000002.bin','000010.bin']);
 assert.equal(seqs[1].id,'00001');
});
test('RELLIS label validation checks point correspondence',async()=>{
 const ok=new Uint32Array([8,17]);
 assert.deepEqual(await parseRellisLabel(new File([ok.buffer],'000000.label'),2),[8,17]);
 await assert.rejects(()=>parseRellisLabel(new File([ok.buffer],'bad.label'),3));
});
test('local tracker persists, expires, and resets on seeking',()=>{
 const cfg={...TRACKING_DEFAULTS,stationarySensor:true,minClusterPoints:1,maxMissed:1};
 const tr=new LocalTracker();
 let out=tr.update([{trackId:0,centroid:[0,0,1],min:[0,0,1],max:[.5,.5,2],dimensions:[.5,.5,1],points:5,velocity:null,speedMps:null,motionStatus:'unknown',motionFrame:'stationary-sensor-assumed',trackHistory:[[0,0,1]],source:'geometric-estimate',trackingStatus:'detected'}],0,0,cfg);
 const id=out[0].trackId;
 out=tr.update([{...out[0],trackId:0,centroid:[1,0,1],trackHistory:[[1,0,1]]}],1,1,cfg);
 assert.equal(out[0].trackId,id);
 assert.equal(out[0].motionStatus,'stationary');
 tr.update([],2,2,cfg);tr.update([],3,3,cfg);
 out=tr.update([{...out[0],trackId:0,centroid:[1.1,0,1],trackHistory:[[1.1,0,1]]}],4,4,cfg);
 assert.notEqual(out[0].trackId,id);
 out=tr.update([{...out[0],trackId:0,centroid:[0,0,1],trackHistory:[[0,0,1]]}],1,1,cfg,true);
 assert.equal(out[0].trackId,1);
});
test('known moving object and uncompensated motion status',()=>{
 const cfg={...TRACKING_DEFAULTS,stationarySensor:true,minClusterPoints:1,movingEnterMps:.5};
 const tr=new LocalTracker();
 let base={trackId:0,centroid:[0,0,1],min:[0,0,1],max:[1,1,2],dimensions:[1,1,1],points:8,velocity:null,speedMps:null,motionStatus:'unknown',motionFrame:'stationary-sensor-assumed',trackHistory:[[0,0,1]],source:'geometric-estimate',trackingStatus:'detected'};
 tr.update([base],0,0,cfg);
 const moved=tr.update([{...base,centroid:[2,0,1],trackHistory:[[2,0,1]]}],1,1,cfg)[0];
 assert.equal(moved.motionStatus,'moving');
 assert.equal(moved.speedMps,2);
 const rel=new LocalTracker().update([base],0,0,{...cfg,stationarySensor:false})[0];
 assert.equal(rel.motionStatus,'unknown');
 assert.equal(rel.speedMps,null);
});
test('static scene under moving sensor requires pose compensation',()=>{
 const cfg={...TRACKING_DEFAULTS,minClusterPoints:1,stationarySensor:false};
 const obj={trackId:0,centroid:[0,0,1],min:[0,0,1],max:[1,1,2],dimensions:[1,1,1],points:8,velocity:null,speedMps:null,motionStatus:'unknown',motionFrame:'sensor-relative-uncompensated',trackHistory:[[0,0,1]],source:'geometric-estimate',trackingStatus:'detected'};
 const out=new LocalTracker().update([obj],0,0,cfg)[0];
 assert.equal(out.motionFrame,'sensor-relative-uncompensated');
});
test('capture timing is separate from playback speed',()=>{
 const seq=groupSequences([new File([new ArrayBuffer(16)],'000010.bin')],5)[0];
 assert.equal(seq.frames[0].timestamp,0);
 assert.equal(seq.assumedHz,5);
});
test('frame cache returns latest requested result under rapid seeking',async()=>{
 const bytes=new Float32Array([1,2,3,.5]).buffer;
 const seq=groupSequences([new File([bytes],'000000.bin'),new File([bytes],'000001.bin')],10)[0];
 const cache=new FrameCache(2);
 const a=cache.load(seq.frames[0]);
 const b=cache.load(seq.frames[1]);
 assert.equal((await b).name,'000001.bin');
 assert.equal((await a).name,'000000.bin');
});
