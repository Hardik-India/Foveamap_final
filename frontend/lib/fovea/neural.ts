import weights from './demo_weights.json' with {type:'json'};
const layers=[{w:weights.w0,b:weights.b0},{w:weights.w1,b:weights.b1},{w:weights.w2,b:weights.b2}];
/** Two-hidden-layer point MLP trained only on procedural scans. No ground-truth label input. */
export function neural(p:{x:number;y:number;z:number;intensity?:number}){let a=[p.x/100,p.y/100,p.z/5,p.intensity??0,Math.hypot(p.x,p.y)/100];for(let k=0;k<3;k++){const {w,b}=layers[k];const out=b.slice();for(let i=0;i<a.length;i++)for(let j=0;j<out.length;j++)out[j]+=a[i]*w[i][j];a=k<2?out.map(v=>Math.max(0,v)):out;}return a.indexOf(Math.max(...a));}
