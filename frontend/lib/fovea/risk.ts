import type { Cell } from './engine.ts';

export type TerrainRiskKey =
  | 'safe'
  | 'caution'
  | 'blocked'
  | 'dynamic'
  | 'unknown';

export type TerrainRisk = {
  key: TerrainRiskKey;
  label: string;
  score: number;
  colour: string;
  reason: string;
};

export const RISK_COLOURS: Record<TerrainRiskKey, string> = {
  safe: '#38d6ad',
  caution: '#e4b35f',
  blocked: '#f47d92',
  dynamic: '#ff8a67',
  unknown: '#a6b4c8',
};

const LABEL_NAMES = ['Drivable', 'Non-drivable', 'Static obstacle', 'Dynamic object'];

function baseReason(cell: Cell) {
  const className = LABEL_NAMES[cell.label] || 'Unknown';
  return `${className}, roughness ${cell.roughness.toFixed(2)} m, height span ${(
    cell.max - cell.min
  ).toFixed(2)} m`;
}

export function classifyTerrainRisk(cell: Cell): TerrainRisk {
  const heightSpan = cell.max - cell.min;

  if (cell.count < 2) {
    return {
      key: 'unknown',
      label: 'Sparse / unknown',
      score: 0.5,
      colour: RISK_COLOURS.unknown,
      reason: baseReason(cell),
    };
  }

  if (cell.label === 3) {
    return {
      key: 'dynamic',
      label: 'Dynamic hazard',
      score: 1,
      colour: RISK_COLOURS.dynamic,
      reason: baseReason(cell),
    };
  }

  if (cell.label === 2 || heightSpan > 0.7 || cell.roughness > 0.24) {
    return {
      key: 'blocked',
      label: 'Blocked',
      score: 0.95,
      colour: RISK_COLOURS.blocked,
      reason: baseReason(cell),
    };
  }

  if (cell.label === 1 || heightSpan > 0.25 || cell.roughness > 0.11) {
    return {
      key: 'caution',
      label: 'Caution',
      score: 0.65,
      colour: RISK_COLOURS.caution,
      reason: baseReason(cell),
    };
  }

  return {
    key: 'safe',
    label: 'Low risk',
    score: 0.15,
    colour: RISK_COLOURS.safe,
    reason: baseReason(cell),
  };
}

export function classifyPlannerReadiness(cell: Cell): TerrainRisk {
  const risk = classifyTerrainRisk(cell);

  if (risk.key === 'safe') {
    return {
      ...risk,
      label: 'Planner-ready',
      score: 0.1,
      reason: `Traversable candidate: ${risk.reason}`,
    };
  }

  if (risk.key === 'caution') {
    return {
      ...risk,
      label: 'Needs slow path',
      score: 0.55,
      reason: `Planner should slow or replan: ${risk.reason}`,
    };
  }

  if (risk.key === 'blocked' || risk.key === 'dynamic') {
    return {
      ...risk,
      label: risk.key === 'dynamic' ? 'Dynamic exclusion' : 'Blocked exclusion',
      score: 1,
      reason: `Planner exclusion: ${risk.reason}`,
    };
  }

  return {
    ...risk,
    label: 'Unknown clearance',
    score: 0.75,
    reason: `Planner needs confirmation: ${risk.reason}`,
  };
}

export function summariseTerrain(
  cells: Cell[],
  classifier: (cell: Cell) => TerrainRisk = classifyTerrainRisk,
) {
  const summary: Record<TerrainRiskKey, number> = {
    safe: 0,
    caution: 0,
    blocked: 0,
    dynamic: 0,
    unknown: 0,
  };

  for (const cell of cells) {
    summary[classifier(cell).key] += 1;
  }

  return summary;
}
