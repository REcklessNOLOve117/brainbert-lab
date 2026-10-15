#!/usr/bin/env python3
"""Export the real BrainBERT demo sample using the training STFT transform.

Dependencies: numpy, scipy. No model or PyTorch inference is performed.
Example:
  python scripts/prepare_data.py --repo /path/to/brainbert
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy import signal


def rounded(array: np.ndarray, decimals: int = 6) -> list:
    # Convert to Python floats only after rounding in float64, avoiding a second
    # float32 representation artifact when serializing JSON.
    return np.round(np.asarray(array, dtype=np.float64), decimals).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path,
        default=Path("/workspace/scratch/4d247005e751/brainbert"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parent.parent / "dist" / "brain-data.json",
    )
    args = parser.parse_args()

    source = args.repo / "notebooks" / "example_wav_1.npy"
    original = np.load(source, allow_pickle=False)
    waveform = original.astype(np.float32)
    fs, nperseg, noverlap, bins, crop = 2048, 400, 350, 40, 10

    # These explicit defaults match scipy.signal.stft in preprocessors/stft.py.
    frequencies, centers, spectrum = signal.stft(
        waveform,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        detrend=False,
        return_onesided=True,
        boundary="zeros",
        padded=True,
        scaling="spectrum",
    )
    magnitude = np.abs(spectrum[:bins])

    # Important: training normalizes each frequency over ALL 206 frames before
    # cropping; the demo notebook instead crops first and uses five frames.
    means = magnitude.mean(axis=-1, keepdims=True)
    stds = magnitude.std(axis=-1, ddof=0, keepdims=True)
    safe_stds = stds.copy()
    safe_stds[safe_stds == 0] = 1.0
    normalized = (magnitude - means) / safe_stds
    if normalized.std() == 0:
        normalized = np.ones_like(normalized)
    normalized = np.nan_to_num(normalized[:, crop:-crop], nan=0.0)
    kept_centers = centers[crop:-crop]
    kept_magnitude = magnitude[:, crop:-crop]

    assert waveform.shape == (10240,)
    assert normalized.T.shape == (186, 40)
    assert np.isfinite(normalized).all()
    assert len(kept_centers) == normalized.shape[1]

    payload = {
        "sample": {
            "fs": fs,
            "n": int(waveform.size),
            "duration": waveform.size / fs,
            "waveform": rounded(waveform),
            "unit": "µV",
            "unitSource": "notebooks/demo.ipynb labels the example voltage axis as Voltage (µV); the .npy file contains no unit metadata.",
        },
        "stft": {
            "time": rounded(kept_centers, 9),
            "frequency": rounded(frequencies[:bins], 8),
            "values": rounded(normalized.T),
            "magnitude": rounded(kept_magnitude.T),
            "shape": [int(normalized.shape[1]), int(normalized.shape[0])],
            "layout": "time × frequency",
            "valueUnit": "z-score (dimensionless)",
            "magnitudeUnit": "STFT magnitude, in input voltage units under spectrum scaling; not squared power",
            "normalizationMean": rounded(means[:, 0]),
            "normalizationStd": rounded(stds[:, 0]),
            "params": {
                "fs": fs,
                "window": "hann",
                "windowPeriodic": True,
                "nperseg": nperseg,
                "noverlap": noverlap,
                "hop": nperseg - noverlap,
                "nfft": nperseg,
                "detrend": False,
                "return_onesided": True,
                "boundary": "zeros",
                "padded": True,
                "scaling": "spectrum",
                "frequencyBinsKept": bins,
                "cropFramesEachSide": crop,
                "normalization": "per-frequency z-score over time, ddof=0, before cropping",
                "fullFrameCount": int(centers.size),
                "fullFrequencyCount": int(frequencies.size),
                "windowDurationSeconds": nperseg / fs,
                "hopDurationSeconds": (nperseg - noverlap) / fs,
                "frequencyStepHz": fs / nperseg,
            },
        },
        "meta": {
            "source": "czlwang/brainbert/notebooks/example_wav_1.npy",
            "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "preprocessor": "czlwang/brainbert/preprocessors/stft.py: STFTPreprocessor.get_stft",
            "sourceDtype": str(original.dtype),
            "calculationDtype": "float32",
            "numpyVersion": np.__version__,
            "scipyVersion": scipy.__version__,
            "waveformDecimals": 6,
            "featureDecimals": 6,
            "containsModelInference": False,
            "notes": [
                "This is the actual repository sample and an actual STFT calculation, not generated signal data.",
                "STFT centers are in seconds relative to sample start, not window start times. Use center ± nperseg/(2*fs) for a selected frame window.",
                "The training preprocessor crops the spectrogram but leaves its returned time array untrimmed. This export trims time consistently with values.",
                "Frequency bins are 0, 5.12, ..., 199.68 Hz. Forty bins do not mean a 40 Hz frequency cutoff.",
                "The notebook plots its last frequency label as 200 Hz; this export preserves the exact FFT frequency of 199.68 Hz.",
                "Training uses abs(STFT), not squared power or logarithmic power. Z-score is computed before cropping, so cropped frequency rows need not have zero mean or unit variance.",
                "The original waveform is float64 and is converted to float32 before preprocessing to match dataset loading.",
                "No pretrained weights, learned embeddings, learned attention, model reconstructions, classification outputs, or fine-tuning outputs are included.",
            ],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.out),
        "bytes": args.out.stat().st_size,
        "shape": payload["stft"]["shape"],
        "timeRange": [float(kept_centers[0]), float(kept_centers[-1])],
        "frequencyRange": [float(frequencies[0]), float(frequencies[bins - 1])],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
