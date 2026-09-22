import { RELLIS_NAMES } from "./sequence.ts";

export type SemanticSource =
  | "none"
  | "rellis-ground-truth"
  | "model-prediction"
  | "heuristic";
export type SemanticPrediction = {
  source: SemanticSource;
  labels: number[];
  names: Record<number, string>;
  modelId?: string;
  checkpointHash?: string;
  pointCount: number;
  featureOrder: "x,y,z,intensity";
  units: "metres";
};

export const FOUR_CLASS_NAMES: Record<number, string> = {
  0: "drivable",
  1: "non-drivable",
  2: "static-obstacle",
  3: "dynamic-object",
};

export function validatePrediction(
  pred: SemanticPrediction,
  pointCount: number,
) {
  if (pred.pointCount !== pointCount || pred.labels.length !== pointCount)
    throw Error(
      "Prediction labels must correspond one-to-one with the current point order.",
    );
  if (
    pred.source === "model-prediction" &&
    (!pred.modelId || !pred.checkpointHash)
  )
    throw Error(
      "Model predictions require modelId and checkpointHash metadata.",
    );
  for (const label of pred.labels)
    if (!Number.isInteger(label) || label < 0)
      throw Error(
        "Prediction labels must be non-negative integer ontology IDs.",
      );
  return pred;
}

export function rellisGroundTruth(
  labels: number[],
  pointCount: number,
): SemanticPrediction {
  return validatePrediction(
    {
      source: "rellis-ground-truth",
      labels,
      names: RELLIS_NAMES,
      pointCount,
      featureOrder: "x,y,z,intensity",
      units: "metres",
    },
    pointCount,
  );
}

/** Build a SemanticPrediction from a backend /track response where semanticSource === 'model-prediction'. */
export function modelPrediction(
  labels: number[],
  pointCount: number,
  modelId: string,
  checkpointHash: string,
): SemanticPrediction {
  return validatePrediction(
    {
      source: "model-prediction",
      labels,
      names: FOUR_CLASS_NAMES,
      modelId,
      checkpointHash,
      pointCount,
      featureOrder: "x,y,z,intensity",
      units: "metres",
    },
    pointCount,
  );
}