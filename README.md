# Systolic Array — FIR Filter Simulation

> A cycle-accurate Python simulation of a weight-stationary systolic array performing FIR filtering — the same parallel architecture used in Google TPUs and real DSP chips. Verified to machine precision against NumPy.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Precision](https://img.shields.io/badge/Max%20Error-2.22e--16-brightgreen?style=flat-square)]()

---

## Overview

Every phone call, music stream, and ECG reading passes through a **FIR (Finite Impulse Response) filter** — a process that removes unwanted noise while preserving the useful signal. On a conventional processor, computing a filter with N taps requires N multiplications done sequentially.

This project implements a **systolic array** — a grid of parallel Processing Elements (PEs) where each PE handles exactly one multiplication per clock cycle. The result: all N multiplications fire simultaneously, producing one filtered output sample per clock tick.

---

## Architecture

### FIR Filter Equation

Each output sample is a weighted sum of the last N input samples:

```
y[n] = h[0]·x[n] + h[1]·x[n-1] + h[2]·x[n-2] + ... + h[N-1]·x[n-(N-1)]
```

### Systolic Execution (16-tap example)

```
PE0  ──  h[0]  × x[n]      ┐
PE1  ──  h[1]  × x[n-1]    │
PE2  ──  h[2]  × x[n-2]    │  All PEs fire simultaneously
 ·                           │  every clock cycle
 ·                           │
PE15 ──  h[15] × x[n-15]   ┘

                  └─── Σ ───► y[n]
```

### Delay Line (Shift Register)

Each PE receives its required delayed sample via a shift register. A new sample enters from the left every clock tick:

```
Clock 1:  [ x0,   0,   0,   0 ]
Clock 2:  [ x1,  x0,   0,   0 ]
Clock 3:  [ x2,  x1,  x0,   0 ]
Clock 4:  [ x3,  x2,  x1,  x0 ]  ←  All PEs fully loaded
```

### Tap Weight Design (Windowed-Sinc Method)

Filter coefficients are derived mathematically in three steps:

| Step | Operation | Purpose |
|------|-----------|---------|
| 1 | `h[k] = sinc(2·fc·(k − mid))` | Ideal low-pass response in time domain |
| 2 | Multiply by Hamming or Blackman window | Suppress spectral leakage and Gibbs ringing |
| 3 | Normalise by sum of weights | Ensure unity DC gain |

Weights are loaded once into the PEs and never change. Only the signal moves.

---

## Features

### Filter Configurations

| Configuration | Taps | Window | Use Case |
|---|---|---|---|
| Low Pass (basic) | 8 | Rectangular | Minimal resource usage |
| Low Pass (main) | 16 | Hamming | General-purpose noise removal |
| Low Pass (sharp) | 32 | Blackman | Maximum stopband attenuation |
| Band Pass | 16 | Hamming | 80–150 Hz isolation |
| High Pass | 16 | Hamming | 200 Hz cutoff, DC blocking |

### Test Signals

| Signal | Description |
|---|---|
| Pure sinusoid | 50 Hz reference tone |
| Multi-tone mix | 50 + 200 + 400 Hz composite |
| Noisy sinusoid | Gaussian white noise at 5 dB SNR |
| Chirp | 10 → 450 Hz frequency sweep |
| Synthetic ECG | Heartbeat waveform with additive noise |

### Outputs

- **Main Dashboard** — 6-panel plot: noise suppression, frequency response, multi-tone separation, PE activity heatmap, tap coefficients, benchmark comparison
- **Systolic Animation** — clock-by-clock GIF of samples flowing through PE cells; active PEs highlighted per cycle
- **Filter Bank Response** — magnitude and phase response of all 5 filters overlaid, showing tap count vs. sharpness trade-off
- **ECG Denoising Demo** — raw and filtered ECG comparison through the 32-tap Blackman LPF

---

## Verification

Every output is compared against `numpy.convolve`. Across all filter configurations and test signals:

```
✓ Match   : True
Max error : 2.22e-16   (machine precision)
```

The systolic array is mathematically equivalent to a direct-form FIR implementation.

---

## Quickstart

### Install dependencies

```bash
pip install numpy matplotlib pillow
```

### Run core verification

```bash
python systolic_array.py
```

Expected output:
```
✓ Match   : True
Max error : 2.22e-16
```

### Verify signal processing layer

```bash
python dsp_signals.py
```

Expected output:
```
SNR before : 5.2 dB
SNR after  : 9.0 dB
Gain       : +3.8 dB
```

### Generate all plots and animation

```bash
python visualizer.py
```

---

## Specifications

| Property | Value |
|---|---|
| Architecture | Weight-stationary 1D systolic array |
| Filter types | FIR — Low Pass, Band Pass, High Pass |
| Tap counts | 8, 16, 32 |
| Window functions | Hamming, Blackman |
| Sample rate | 1000 Hz |
| Default cutoff | 100 Hz |
| Verification error | < 2.22e-16 |
| Python version | 3.10+ |
| Dependencies | NumPy, Matplotlib, Pillow |

---

## Background

### Why systolic arrays?

Conventional processors fetch operands from memory for every computation. Memory access consumes roughly 100× more energy than arithmetic. Systolic arrays eliminate this bottleneck — each PE passes its result directly to its neighbour, so data flows through the array without touching memory between operations. At data centre scale, this translates into measurable reductions in energy consumption and CO₂ emissions.

This architecture directly underlies Google's Tensor Processing Units (TPUs) and is widely deployed in FPGA and ASIC DSP implementations.

### Relevance of this simulation

This is a cycle-accurate behavioural simulation of a weight-stationary 1D systolic array. The PE activity matrix provides clock-level visibility into data flow and utilisation — making it a useful reference for understanding how hardware implementations behave before committing to RTL design.

---

## Project Structure

```
.
├── systolic_array.py   # Core PE logic and array execution engine
├── dsp_signals.py      # Signal generation and SNR measurement
├── visualizer.py       # Dashboard, animation, and filter bank plots
└── README.md
```

---

## Built With

[Python](https://python.org) · [NumPy](https://numpy.org) · [Matplotlib](https://matplotlib.org) · [Pillow](https://python-pillow.org)
