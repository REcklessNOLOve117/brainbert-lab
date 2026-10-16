// Small, deterministic teaching calculations. These are not pretrained weights.
export const POPT_SHAPES = Object.freeze({ input: 768, hidden: 512, heads: 8, head: 64, layers: 6, ffn: 2048 });

export function seededRandom(seed = 1) {
  let value = seed >>> 0;
  return () => {
    value = (Math.imul(value, 1664525) + 1013904223) >>> 0;
    return value / 4294967296;
  };
}

/** Exact sinusoidal table formula used in MultiSubjBrainPositionalEncoding. */
export function sinusoidal(position, dimension = 128) {
  if (!Number.isInteger(position) || position < 0 || position >= 5000) throw new RangeError('Position must be an integer in [0,4999].');
  if (dimension < 2 || dimension % 2) throw new RangeError('Encoding dimension must be positive and even.');
  return Array.from({ length: dimension }, (_, i) => {
    const phase = position * Math.pow(10000, -(2 * Math.floor(i / 2)) / dimension);
    return i % 2 === 0 ? Math.sin(phase) : Math.cos(phase);
  });
}

export function populationPosition(coords, sequenceId = 0, hidden = 512) {
  if (hidden % 8 || coords.length !== 3) throw new RangeError('Expected three coordinates and hidden divisible by 8.');
  const quarter = hidden / 4;
  return [...coords, sequenceId].flatMap(position => sinusoidal(position, quarter));
}

export function layerNorm(vector, epsilon = 1e-5) {
  const mean = vector.reduce((sum, x) => sum + x, 0) / vector.length;
  const variance = vector.reduce((sum, x) => sum + (x - mean) ** 2, 0) / vector.length;
  return vector.map(x => (x - mean) / Math.sqrt(variance + epsilon));
}

export function softmax(values) {
  const maximum = Math.max(...values);
  const exp = values.map(value => Math.exp(value - maximum));
  const total = exp.reduce((sum, x) => sum + x, 0);
  return exp.map(x => x / total);
}

export function binaryCrossEntropy(logit, target) {
  if (target !== 0 && target !== 1) throw new RangeError('BCE target must be 0 or 1.');
  return Math.max(logit, 0) - logit * target + Math.log1p(Math.exp(-Math.abs(logit)));
}

/** The repo ignores 0, uses 1 for a replacement, and maps 2 (kept) to target 0. */
export function replacementLoss(logits, internalLabels) {
  const chosen = internalLabels.map((label, i) => ({ label, logit: logits[i], index: i }))
    .filter(item => item.label === 1 || item.label === 2)
    .map(item => ({ ...item, target: item.label === 1 ? 1 : 0 }));
  const terms = chosen.map(item => ({ ...item, loss: binaryCrossEntropy(item.logit, item.target) }));
  return { count: terms.length, loss: terms.length ? terms.reduce((sum, item) => sum + item.loss, 0) / terms.length : null, terms };
}

function randomMatrix(rows, columns, seed) {
  const random = seededRandom(seed);
  const scale = Math.sqrt(3 / rows);
  return Array.from({ length: columns }, () => Float64Array.from({ length: rows }, () => (random() * 2 - 1) * scale));
}

function multiply(vector, matrix) {
  return matrix.map(column => {
    let result = 0;
    for (let i = 0; i < vector.length; i += 1) result += vector[i] * column[i];
    return result;
  });
}

let inputProjection;
const headMatrices = new Map();
const projectionCache = new Map();

function projectedToken(index) {
  if (!projectionCache.has(index)) {
    inputProjection ??= randomMatrix(768, 512, 17017);
    const random = seededRandom(310 + index * 43);
    const raw = index === 0 ? Array(768).fill(1) : Array.from({ length: 768 }, (_, dimension) =>
      0.7 * Math.sin((dimension + 1) * 0.027 + index * 0.39) + (random() - 0.5) * 0.8);
    projectionCache.set(index, multiply(raw, inputProjection));
  }
  return projectionCache.get(index);
}

/** Same token/PE dimensions as PopT; fixed random weights, one head only, no six-layer inference. */
export function populationAttention(coordinates, head = 0, splitSequence = false, usePosition = true) {
  if (!Number.isInteger(head) || head < 0 || head > 7) throw new RangeError('Head must be in [0,7].');
  const count = coordinates.length;
  const positions = [populationPosition([0, 0, 0], 0), ...coordinates.map((coords, i) =>
    populationPosition(coords, splitSequence && i >= Math.floor(count / 2) ? 1 : 0))];
  const inputs = positions.map((position, index) => layerNorm(projectedToken(index).map((value, d) => value + (usePosition ? position[d] : 0))));
  if (!headMatrices.has(head)) headMatrices.set(head, {
    q: randomMatrix(512, 64, 701 + head * 101),
    k: randomMatrix(512, 64, 1103 + head * 103),
    v: randomMatrix(512, 64, 1709 + head * 107),
  });
  const matrices = headMatrices.get(head);
  const queries = inputs.map(input => multiply(input, matrices.q));
  const keys = inputs.map(input => multiply(input, matrices.k));
  const values = inputs.map(input => multiply(input, matrices.v));
  const scores = queries.map(query => keys.map(key => query.reduce((sum, value, d) => sum + value * key[d], 0) / Math.sqrt(64)));
  const weights = scores.map(softmax);
  const contexts = weights.map(row => Array.from({ length: 64 }, (_, d) => row.reduce((sum, weight, j) => sum + weight * values[j][d], 0)));
  return { inputs, positions, queries, keys, values, scores, weights, contexts };
}

export function defaultCoordinates(count) {
  return Array.from({ length: count }, (_, i) => [80 + (i * 37) % 260, 90 + (i * 53) % 240, 100 + (i * 29) % 220]);
}
