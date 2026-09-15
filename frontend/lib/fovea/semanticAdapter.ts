import {RELLIS_NAMES} from './sequence.ts';

export type SemanticSource='none'|'rellis-ground-truth'|'model-prediction'|'heuristic';
export type SemanticPrediction={
  source:SemanticSource;
  labels:number[];
  names:Record<number,string>;
  modelId?:string;
  checkpointHash?:string;
  pointCount:number;
  featureOrder:'x,y,z,intensity';
  units:'metres';
};

export function validatePrediction(pred:SemanticPrediction,pointCount:number){
  if(pred.pointCount!==pointCount||pred.labels.length!==pointCount)throw Error('Prediction labels must correspond one-to-one with the current point order.');
  if(pred.source==='model-prediction'&&(!pred.modelId||!pred.checkpointHash))throw Error('Model predictions require modelId and checkpointHash metadata.');
  for(const label of pred.labels)if(!Number.isInteger(label)||label<0)throw Error('Prediction labels must be non-negative integer ontology IDs.');
  return pred;
}

export function rellisGroundTruth(labels:number[],pointCount:number):SemanticPrediction{
  return validatePrediction({source:'rellis-ground-truth',labels,names:RELLIS_NAMES,pointCount,featureOrder:'x,y,z,intensity',units:'metres'},pointCount);
}
