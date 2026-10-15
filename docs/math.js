/**
 * BrainBERT teaching math, with no third-party runtime dependencies.
 *
 * IMPORTANT: the input projection and Q/K/V parameters below are fixed-seed
 * RANDOM TEACHING PARAMETERS, never parameters from stft_large_pretrained.pth.
 * makeMaskedExample's prediction is controlled synthetic noise around target;
 * it is not a model reconstruction. Only stftWaveform performs the actual
 * deterministic signal transform used by the repository's STFT preprocessor.
 */

const INPUT_DIM = 40;
const MODEL_DIM = 768;
const HEAD_DIM = 64;
const INPUT_SEED = 0xb40b0768;
const ATTENTION_SEED = 0xa776b37;
const EPSILON = 1e-5;

export const teachingParameterInfo = Object.freeze({
  pretrained: false,
  inputProjection: 'Fixed-seed random linear weights; zero bias, 40 → 768',
  attention: 'Fixed-seed random Q/K/V weights per head; zero bias, 768 → 64',
  layerNorm: 'Unit scale, zero bias, population variance, epsilon = 1e-5',
  dropout: 'Disabled (evaluation mode)',
  reconstruction: 'Target plus deterministic synthetic noise, not model output',
});

function randomGenerator(seed) {
  let state = Number(seed) >>> 0;
  return function random() {
    state = (state + 0x6d2b79f5) | 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value ^= value + Math.imul(value ^ (value >>> 7), 61 | value);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianGenerator(random) {
  let spare = null;
  return function normal() {
    if (spare !== null) {
      const value = spare;
      spare = null;
      return value;
    }
    // 1 - random() is in (0, 1], keeping log finite.
    const radius = Math.sqrt(-2 * Math.log(1 - random()));
    const angle = 2 * Math.PI * random();
    spare = radius * Math.sin(angle);
    return radius * Math.cos(angle);
  };
}

function matrixDimensions(values, requiredWidth) {
  if (!Array.isArray(values)) throw new TypeError('Expected an array of rows');
  const width = values.length ? values[0].length : requiredWidth;
  if (requiredWidth !== undefined && width !== requiredWidth) {
    throw new RangeError(`Expected ${requiredWidth} features per row`);
  }
  for (const row of values) {
    if (!row || row.length !== width) throw new RangeError('Matrix must be rectangular');
    for (const value of row) {
      if (!Number.isFinite(Number(value))) throw new RangeError('Matrix contains a non-finite number');
    }
  }
  return { rows: values.length, columns: width };
}

function cloneMatrix(values) {
  return values.map((row) => Float64Array.from(row));
}

let inputProjection = null;
const attentionProjections = new Map();

function getInputProjection() {
  if (!inputProjection) {
    const random = randomGenerator(INPUT_SEED);
    const bound = 1 / Math.sqrt(INPUT_DIM);
    inputProjection = new Float64Array(INPUT_DIM * MODEL_DIM);
    for (let i = 0; i < inputProjection.length; i++) {
      inputProjection[i] = (2 * random() - 1) * bound;
    }
  }
  return inputProjection;
}

/**
 * Convert real continuous frequency features to teaching input embeddings.
 * No token ID lookup: each frame is multiplied by a 40 × 768 matrix.
 * stage: 'linear' = Wx, 'position' = Wx + sinusoidal PE,
 *        'norm' = LayerNorm(Wx + PE), with evaluation-mode dropout disabled.
 * Returns one 768-dimensional Float64Array per time frame.
 */
export function makeInputEmbeddings(values, stage = 'norm') {
  matrixDimensions(values, INPUT_DIM);
  if (!['linear', 'position', 'norm'].includes(stage)) {
    throw new RangeError("stage must be 'linear', 'position', or 'norm'");
  }
  const projection = getInputProjection();
  const frequencies = new Float64Array(MODEL_DIM / 2);
  for (let pair = 0; pair < frequencies.length; pair++) {
    frequencies[pair] = Math.exp((-Math.log(10000) * (2 * pair)) / MODEL_DIM);
  }

  return values.map((frame, time) => {
    const result = new Float64Array(MODEL_DIM);
    for (let feature = 0; feature < INPUT_DIM; feature++) {
      const value = Number(frame[feature]);
      const offset = feature * MODEL_DIM;
      for (let dim = 0; dim < MODEL_DIM; dim++) {
        result[dim] += value * projection[offset + dim];
      }
    }
    if (stage === 'linear') return result;

    for (let pair = 0; pair < frequencies.length; pair++) {
      const angle = time * frequencies[pair];
      result[2 * pair] += Math.sin(angle);
      result[2 * pair + 1] += Math.cos(angle);
    }
    if (stage === 'position') return result;

    let mean = 0;
    for (const value of result) mean += value;
    mean /= MODEL_DIM;
    let variance = 0;
    for (const value of result) variance += (value - mean) ** 2;
    variance /= MODEL_DIM;
    const inverseStd = 1 / Math.sqrt(variance + EPSILON);
    for (let dim = 0; dim < MODEL_DIM; dim++) {
      result[dim] = (result[dim] - mean) * inverseStd;
    }
    return result;
  });
}

function getAttentionProjection(head) {
  if (!attentionProjections.has(head)) {
    const random = randomGenerator((ATTENTION_SEED + Math.imul(head + 1, 0x9e3779b9)) >>> 0);
    // Variance 1 / MODEL_DIM provides readable softmax variation in the demo.
    // This is a teaching initialization, not PyTorch's pretrained MHA tensor.
    const bound = Math.sqrt(3 / MODEL_DIM);
    const weights = Array.from({ length: 3 }, () => new Float64Array(MODEL_DIM * HEAD_DIM));
    for (const matrix of weights) {
      for (let i = 0; i < matrix.length; i++) matrix[i] = (2 * random() - 1) * bound;
    }
    attentionProjections.set(head, weights);
  }
  return attentionProjections.get(head);
}

/**
 * Compute one real scaled-dot-product attention head using random teaching
 * projections. Q/K/V each have shape T × 64; weights have shape T × T.
 * padding masks the last N KEY columns; query rows remain present, matching
 * src_key_padding_mask semantics. There is no causal mask.
 * Temperature = 1 matches the standard sqrt(64) = 8 scaling.
 * When all keys are padded, returns zero weights (UI guard against NaN).
 */
export function computeAttention(embeddings, head = 0, { temperature = 1, padding = 0 } = {}) {
  const { rows: count } = matrixDimensions(embeddings, MODEL_DIM);
  if (!Number.isInteger(head) || head < 0 || head >= 12) {
    throw new RangeError('head must be an integer from 0 through 11');
  }
  if (!Number.isFinite(temperature) || temperature <= 0) {
    throw new RangeError('temperature must be positive');
  }
  const paddedKeys = Math.max(0, Math.min(count, Math.floor(Number(padding) || 0)));
  const validKeys = count - paddedKeys;
  const matrices = getAttentionProjection(head);
  const projections = matrices.map((matrix) => embeddings.map((frame) => {
    const projected = new Float64Array(HEAD_DIM);
    for (let dim = 0; dim < MODEL_DIM; dim++) {
      const value = Number(frame[dim]);
      const offset = dim * HEAD_DIM;
      for (let target = 0; target < HEAD_DIM; target++) {
        projected[target] += value * matrix[offset + target];
      }
    }
    return projected;
  }));
  const [Q, K, V] = projections;
  const scale = 1 / (Math.sqrt(HEAD_DIM) * temperature);
  const weights = Q.map((query) => {
    const row = new Float64Array(count);
    if (!validKeys) return row;
    let maximum = -Infinity;
    for (let key = 0; key < validKeys; key++) {
      let dot = 0;
      for (let dim = 0; dim < HEAD_DIM; dim++) dot += query[dim] * K[key][dim];
      row[key] = dot * scale;
      if (row[key] > maximum) maximum = row[key];
    }
    // Stable softmax over valid key columns only; remaining columns stay zero.
    let total = 0;
    for (let key = 0; key < validKeys; key++) {
      row[key] = Math.exp(row[key] - maximum);
      total += row[key];
    }
    for (let key = 0; key < validKeys; key++) row[key] /= total;
    return row;
  });
  return { Q, K, V, weights, headDim: HEAD_DIM };
}

function maskedIntervals(length, maximumSpan, probability, random) {
  const intervals = [];
  const validStartCount = Math.max(0, length - maximumSpan);
  // Same valid starts and gap rule as util/mask_utils.py. RNG values are
  // deterministic JS values and are not identical to Python random's stream.
  for (let start = 0; start < validStartCount; start++) {
    if (random() < probability && (!intervals.length || intervals.at(-1)[1] < start)) {
      const span = 1 + Math.floor(random() * maximumSpan);
      intervals.push([start, start + span]);
    }
  }
  return { intervals, validStartCount };
}

/**
 * Reproduce the fixed-mask mechanics, then provide SYNTHETIC predictions for
 * explaining the actual masked L1 + alpha * high-amplitude L1 calculation.
 *
 * mode 'time': 1..5 time frames; 'frequency'/'freq': 1..2 frequency bins;
 * 'both': time followed by frequency, with mask labels taking their union.
 * Each selected interval has 10% unchanged / 10% copied / 80% zeroed inputs.
 * Unchanged selected cells still contribute to the loss.
 * prediction = target + error * deterministic standard-normal noise;
 * it is NOT a function learned from the masked input.
 *
 * For an empty high-amplitude subset highL1 is null and its extra contribution
 * is treated as zero (a UI guard; the repository's empty torch.mean is NaN).
 * No selected cells => l1, highL1, and loss are null.
 */
export function makeMaskedExample(values, {
  mode = 'both', probability = 0.1, seed = 1, error = 0.5, alpha = 2,
} = {}) {
  const { rows: times, columns: frequencies } = matrixDimensions(values, INPUT_DIM);
  if (!['time', 'frequency', 'freq', 'both', 'none'].includes(mode)) {
    throw new RangeError("mode must be 'time', 'frequency', 'freq', 'both', or 'none'");
  }
  if (![probability, error, alpha].every(Number.isFinite)) throw new RangeError('Options must be finite');
  const chance = Math.min(1, Math.max(0, probability));
  const noiseScale = Math.max(0, error);
  const random = randomGenerator(seed);
  const target = cloneMatrix(values);
  const input = cloneMatrix(values);
  const mask = Array.from({ length: times }, () => new Uint8Array(frequencies));

  if (mode === 'time' || mode === 'both') {
    const { intervals, validStartCount } = maskedIntervals(times, 5, chance, random);
    // Mark every selected cell independently of how its input is corrupted.
    for (const [start, end] of intervals) {
      for (let time = start; time < end; time++) mask[time].fill(1);
    }
    for (const [start, end] of intervals) {
      const dice = random();
      if (dice < 0.1) continue;
      if (dice < 0.2) {
        const replacement = Math.floor(random() * validStartCount);
        for (let time = start; time < end; time++) input[time].set(target[replacement + time - start]);
      } else {
        for (let time = start; time < end; time++) input[time].fill(0);
      }
    }
  }

  if (mode === 'frequency' || mode === 'freq' || mode === 'both') {
    const { intervals, validStartCount } = maskedIntervals(frequencies, 2, chance, random);
    for (const [start, end] of intervals) {
      for (let time = 0; time < times; time++) mask[time].fill(1, start, end);
    }
    for (const [start, end] of intervals) {
      const dice = random();
      if (dice < 0.1) continue;
      if (dice < 0.2) {
        const replacement = Math.floor(random() * validStartCount);
        // Like fixed_mask_inputs, frequency replacement uses ORIGINAL target,
        // so it may overwrite part of a previous time corruption.
        for (let time = 0; time < times; time++) {
          for (let frequency = start; frequency < end; frequency++) {
            input[time][frequency] = target[time][replacement + frequency - start];
          }
        }
      } else {
        for (let time = 0; time < times; time++) input[time].fill(0, start, end);
      }
    }
  }

  // Separate RNG keeps the controlled predictions fixed while mask settings
  // change, allowing the reader to inspect which positions enter the loss.
  const normal = gaussianGenerator(randomGenerator((Number(seed) ^ 0xc0ffee42) >>> 0));
  const prediction = Array.from({ length: times }, () => new Float64Array(frequencies));
  const errors = Array.from({ length: times }, () => new Float64Array(frequencies));
  const high = Array.from({ length: times }, () => new Uint8Array(frequencies));
  let sum = 0, highSum = 0, count = 0, highCount = 0;
  for (let time = 0; time < times; time++) {
    for (let frequency = 0; frequency < frequencies; frequency++) {
      prediction[time][frequency] = target[time][frequency] + noiseScale * normal();
      const difference = Math.abs(target[time][frequency] - prediction[time][frequency]);
      errors[time][frequency] = difference;
      if (mask[time][frequency]) {
        count++;
        sum += difference;
        if (Math.abs(target[time][frequency]) > 1) {
          high[time][frequency] = 1;
          highCount++;
          highSum += difference;
        }
      }
    }
  }
  const l1 = count ? sum / count : null;
  const highL1 = highCount ? highSum / highCount : null;
  const loss = count ? l1 + alpha * (highL1 ?? 0) : null;
  return { input, target, prediction, mask, high, errors, loss, l1, highL1, count, highCount };
}

const STFT_SIZE = 400;
const STFT_HOP = 50;
const STFT_BINS = 40;
const STFT_CROP = 10;
let dftTables = null;

function getDFTTables() {
  if (!dftTables) {
    const cosine = new Float64Array(STFT_BINS * STFT_SIZE);
    const sine = new Float64Array(STFT_BINS * STFT_SIZE);
    const window = new Float64Array(STFT_SIZE);
    let windowSum = 0;
    for (let sample = 0; sample < STFT_SIZE; sample++) {
      // scipy.signal.get_window('hann', 400, fftbins=True): periodic Hann.
      window[sample] = 0.5 - 0.5 * Math.cos(2 * Math.PI * sample / STFT_SIZE);
      windowSum += window[sample];
    }
    for (let frequency = 0; frequency < STFT_BINS; frequency++) {
      const offset = frequency * STFT_SIZE;
      for (let sample = 0; sample < STFT_SIZE; sample++) {
        const angle = 2 * Math.PI * frequency * sample / STFT_SIZE;
        cosine[offset + sample] = Math.cos(angle) * window[sample] / windowSum;
        sine[offset + sample] = -Math.sin(angle) * window[sample] / windowSum;
      }
    }
    dftTables = { cosine, sine, windowSum };
  }
  return dftTables;
}

/**
 * Real STFT, not a heatmap approximation. The return value has the same fields
 * as brain-data.json's stft object and uses T × 40 row layout.
 *
 * Matches signal.stft(float32(waveform), fs, window='hann', nperseg=400,
 * noverlap=350, return_onesided=True, boundary='zeros', padded=True,
 * scaling='spectrum'), abs -> per-frequency zscore over FULL frames -> crop.
 * Direct DFT arithmetic uses float64; small FFT/float32 rounding differences
 * from SciPy are expected. For 10240 samples it yields 206 full / 186 kept
 * frames, with centers 0.244140625 through 4.7607421875 s at fs = 2048.
 */
export function stftWaveform(waveform, fs = 2048) {
  if (!waveform || waveform.length < STFT_SIZE) {
    throw new RangeError('STFT demo requires at least 400 waveform samples');
  }
  if (!Number.isFinite(fs) || fs <= 0) throw new RangeError('fs must be positive');
  const samples = Float32Array.from(waveform);
  for (const sample of samples) {
    if (!Number.isFinite(sample)) throw new RangeError('Waveform contains a non-finite number');
  }
  const fullFrames = 1 + Math.ceil(samples.length / STFT_HOP);
  if (fullFrames <= 2 * STFT_CROP) throw new RangeError('Waveform is too short after cropping');
  const keptFrames = fullFrames - 2 * STFT_CROP;
  const { cosine, sine } = getDFTTables();
  const magnitudeFull = Array.from({ length: fullFrames }, () => new Float64Array(STFT_BINS));
  const segment = new Float64Array(STFT_SIZE);
  for (let frame = 0; frame < fullFrames; frame++) {
    const start = frame * STFT_HOP - STFT_SIZE / 2;
    for (let sample = 0; sample < STFT_SIZE; sample++) {
      const source = start + sample;
      segment[sample] = source >= 0 && source < samples.length ? samples[source] : 0;
    }
    for (let frequency = 0; frequency < STFT_BINS; frequency++) {
      const offset = frequency * STFT_SIZE;
      let real = 0, imaginary = 0;
      for (let sample = 0; sample < STFT_SIZE; sample++) {
        real += segment[sample] * cosine[offset + sample];
        imaginary += segment[sample] * sine[offset + sample];
      }
      magnitudeFull[frame][frequency] = Math.hypot(real, imaginary);
    }
  }

  const means = new Float64Array(STFT_BINS);
  const stds = new Float64Array(STFT_BINS);
  for (let frequency = 0; frequency < STFT_BINS; frequency++) {
    let sum = 0;
    for (let frame = 0; frame < fullFrames; frame++) sum += magnitudeFull[frame][frequency];
    means[frequency] = sum / fullFrames;
    let variance = 0;
    for (let frame = 0; frame < fullFrames; frame++) {
      variance += (magnitudeFull[frame][frequency] - means[frequency]) ** 2;
    }
    stds[frequency] = Math.sqrt(variance / fullFrames);
  }
  // Mirror the preprocessor's unusual all-zero fallback (ones, not zeros).
  const allConstant = stds.every((value) => value === 0);
  const values = new Array(keptFrames);
  const magnitude = new Array(keptFrames);
  const time = new Array(keptFrames);
  for (let kept = 0; kept < keptFrames; kept++) {
    const frame = kept + STFT_CROP;
    const normalized = new Float64Array(STFT_BINS);
    for (let frequency = 0; frequency < STFT_BINS; frequency++) {
      normalized[frequency] = allConstant ? 1 :
        (magnitudeFull[frame][frequency] - means[frequency]) / (stds[frequency] || 1);
    }
    values[kept] = normalized;
    magnitude[kept] = magnitudeFull[frame].slice();
    time[kept] = frame * STFT_HOP / fs;
  }
  const frequency = Array.from({ length: STFT_BINS }, (_, index) => index * fs / STFT_SIZE);
  return {
    time,
    frequency,
    values,
    magnitude,
    shape: [keptFrames, STFT_BINS],
    layout: 'time × frequency',
    valueUnit: 'z-score (dimensionless)',
    magnitudeUnit: 'STFT magnitude, in input voltage units under spectrum scaling; not squared power',
    normalizationMean: means,
    normalizationStd: stds,
    params: {
      fs,
      window: 'hann',
      windowPeriodic: true,
      nperseg: STFT_SIZE,
      noverlap: STFT_SIZE - STFT_HOP,
      hop: STFT_HOP,
      nfft: STFT_SIZE,
      detrend: false,
      return_onesided: true,
      boundary: 'zeros',
      padded: true,
      scaling: 'spectrum',
      frequencyBinsKept: STFT_BINS,
      cropFramesEachSide: STFT_CROP,
      normalization: 'per-frequency z-score over time, ddof=0, before cropping',
      fullFrameCount: fullFrames,
      fullFrequencyCount: STFT_SIZE / 2 + 1,
      windowDurationSeconds: STFT_SIZE / fs,
      hopDurationSeconds: STFT_HOP / fs,
      frequencyStepHz: fs / STFT_SIZE,
    },
  };
}
