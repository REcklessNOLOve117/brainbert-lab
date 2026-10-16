import test from 'node:test';
import assert from 'node:assert/strict';
import { sinusoidal, populationPosition, binaryCrossEntropy, replacementLoss,
  populationAttention, defaultCoordinates } from '../docs/popt-math.js';

const close = (actual, expected, tolerance = 1e-11) =>
  assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);

test('coordinates occupy three independent 128-d blocks; group ID occupies the fourth', () => {
  const baseline = populationPosition([0, 0, 0], 0);
  assert.equal(baseline.length, 512);
  baseline.forEach((value, i) => assert.equal(value, i % 2));
  const moved = populationPosition([1, 0, 0], 0);
  close(moved[0], 0.8414709848078965);
  close(moved[1], 0.5403023058681398);
  assert.deepEqual(moved.slice(128), baseline.slice(128));
  const groupB = populationPosition([0, 0, 0], 1);
  assert.deepEqual(groupB.slice(0, 384), baseline.slice(0, 384));
  close(groupB[384], moved[0]);
  assert.throws(() => sinusoidal(-1), RangeError);
  assert.throws(() => sinusoidal(5000), RangeError);
});

test('BCE is stable at extreme logits and uses the official replacement-label mapping', () => {
  close(binaryCrossEntropy(0, 0), Math.log(2));
  close(binaryCrossEntropy(0, 1), Math.log(2));
  close(binaryCrossEntropy(1000, 0), 1000);
  close(binaryCrossEntropy(-1000, 1), 1000);
  const selected = replacementLoss([1000, 0, 0], [0, 1, 2]);
  assert.equal(selected.count, 2);
  assert.deepEqual(selected.terms.map(x => [x.index, x.target]), [[1, 1], [2, 0]]);
  close(selected.loss, Math.log(2));
  assert.equal(replacementLoss([0], [0]).loss, null);
});

test('attention includes CLS, normalizes every key row, and computes a weighted value sum', () => {
  const result = populationAttention(defaultCoordinates(4));
  assert.equal(result.inputs.length, 5);
  assert.equal(result.inputs[0].length, 512);
  assert.equal(result.queries[0].length, 64);
  for (const row of result.weights) {
    assert.equal(row.length, 5);
    close(row.reduce((a, b) => a + b, 0), 1);
    assert.ok(row.every(x => Number.isFinite(x) && x > 0 && x <= 1));
  }
  const expected = result.weights[0].reduce((sum, weight, key) => sum + weight * result.values[key][0], 0);
  close(result.contexts[0][0], expected);
  assert.deepEqual(result, populationAttention(defaultCoordinates(4)));
});

test('changing a head, position, group ID, or electrode count changes the actual computation', () => {
  const coords = defaultCoordinates(4);
  const baseline = populationAttention(coords);
  const moved = coords.map(x => [...x]);
  moved[0][0] += 1;
  const variants = [populationAttention(coords, 1), populationAttention(moved),
    populationAttention(coords, 0, true), populationAttention(coords, 0, false, false)];
  for (const result of variants) assert.notDeepEqual(result.weights, baseline.weights);
  const larger = populationAttention(defaultCoordinates(12));
  assert.equal(larger.weights.length, 13);
  larger.weights.forEach(row => close(row.reduce((a, b) => a + b, 0), 1));
});
