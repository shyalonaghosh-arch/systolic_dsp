# Systolic Array — FIR Filter Simulation

A Python simulation of a **systolic array** performing **FIR filtering** — the same parallel hardware architecture used inside Google TPUs and real DSP chips, built from scratch and verified to machine precision.

---

## What is this project?

Every time you make a phone call, listen to music, or get an ECG reading, your signal passes through a **filter** — a mathematical process that removes unwanted noise and keeps the useful information.

The most common type is a **FIR (Finite Impulse Response) filter**. It works by taking a weighted average of the last N signal samples to smooth out noise.

The challenge: doing this on a normal processor means N multiplications done **one at a time** — slow and power hungry.

This project solves that using a **systolic array** — a grid of simple processors (called Processing Elements or PEs) that each do one multiplication simultaneously, every clock cycle. The result: all N multiplications done **in parallel**, producing one filtered output sample per clock tick.

---

## The Math (don't worry, it's simpler than it looks)

A FIR filter computes each output sample like this:

```
y[n] = h[0]·x[n] + h[1]·x[n-1] + h[2]·x[n-2] + ... + h[N-1]·x[n-(N-1)]
```

In plain English:
- `x[n]` is the current input sample
- `x[n-1]` is the previous sample, `x[n-2]` is two steps ago, and so on
- `h[0], h[1]...` are fixed weights (tap coefficients) that shape the filter
- `y[n]` is the filtered output

Each term in this sum is **independent** — it doesn't depend on any other term. So instead of computing them one after another, the systolic array computes all of them **at the same time**:

```
PE0  →  h[0] × x[n]      ┐
PE1  →  h[1] × x[n-1]    │  all firing simultaneously
PE2  →  h[2] × x[n-2]    │  every clock cycle
...                        │
PE15 →  h[15] × x[n-15]  ┘

Add all results → y[n] ✓
```

---

## How the tap weights are calculated

The weights `h[0]...h[N-1]` are not guessed — they are mathematically derived using the **windowed-sinc method**:

**Step 1 — Ideal filter (sinc function)**
The ideal low-pass filter in the time domain is a sinc function:
```
h[k] = sinc(2 · fc · (k - mid))
```
Where `fc` is the cutoff frequency normalised to Nyquist, and `mid` centres the filter for zero phase distortion.

**Step 2 — Apply a window**
The raw sinc goes on forever, so we multiply it by a window function (Hamming or Blackman) that smoothly tapers to zero at the edges. This removes ringing and spectral leakage.

**Step 3 — Normalise**
Divide all weights by their sum so a constant signal passes through unchanged (DC gain = 1).

The resulting weights are loaded permanently into the PEs — one weight per PE — and never change during operation. Only the signal moves.

---

## The Delay Line — how each PE gets the right sample

PE0 needs the current sample `x[n]`, PE1 needs `x[n-1]`, PE2 needs `x[n-2]` and so on. This is handled by a **shift register**:

```
Clock 1:  [x0,  0,   0,   0 ]
Clock 2:  [x1,  x0,  0,   0 ]
Clock 3:  [x2,  x1,  x0,  0 ]
Clock 4:  [x3,  x2,  x1,  x0]  ← all PEs have what they need
```

Every clock tick, a new sample enters from the left and everything shifts right. Each PE reads from its fixed position — always getting the exact delayed sample it needs.

---

## What this simulation includes

**Five filter configurations**
- 8-tap Low Pass Filter (basic, fewest PEs)
- 16-tap Low Pass Filter (main demo, Hamming window)
- 32-tap Low Pass Filter (sharpest cutoff, Blackman window)
- 16-tap Band Pass Filter (80–150 Hz)
- 16-tap High Pass Filter (200 Hz cutoff)

**Test signals**
- Pure 50 Hz sinusoid
- Multi-tone mix (50 + 200 + 400 Hz)
- Noisy sinusoid (Gaussian white noise at 5 dB SNR)
- Chirp (frequency sweep 10 → 450 Hz)
- Synthetic ECG (heartbeat waveform with noise)

**Verification**
Every output is compared against NumPy's built-in convolution. Maximum error across all tests: **2.22e-16** — machine precision. The systolic array is mathematically identical to a real FIR filter.

---

## Outputs

**Main Dashboard** — 6 panels showing noise suppression, frequency response, multi-tone separation, PE activity heatmap, tap coefficients, and benchmark comparison.

**Systolic Animation** — clock-by-clock GIF of samples flowing through PE cells. Active PEs light up green every cycle.

**Filter Bank Response** — magnitude and phase response of all 5 filters plotted together, showing the trade-off between tap count and sharpness.

**ECG Denoising Demo** — synthetic ECG signal before and after filtering through a 32-tap Blackman windowed LPF via the systolic array.

---

## Why does this matter?

**For hardware designers and DSP engineers:**
This is a cycle-accurate behavioural simulation of a weight-stationary 1D systolic array. The architecture directly maps to FPGA and ASIC implementations. The PE activity matrix provides cycle-level visibility into data flow and utilisation.

**For beginners:**
Think of it like an assembly line. Each worker (PE) has one fixed job — multiply one number. The product (signal) moves along the line and gets processed by every worker simultaneously. At the end, all the partial results are added up. Fast, efficient, and the same idea behind Google's AI chips.

**For the environment:**
Systolic arrays are energy efficient because data flows directly PE to PE — no repeated fetches from memory. Memory access consumes 100x more power than computation, so minimising it dramatically reduces energy use. At data centre scale this translates to measurable reductions in CO₂ emissions.

---

## Quickstart

**Install**
```bash
pip install numpy matplotlib pillow
```

**Verify the core engine**
```bash
python systolic_array.py
```
```
✓ Match   : True
Max error : 2.22e-16
```

**Verify signal layer**
```bash
python dsp_signals.py
```
```
SNR before : 5.2 dB
SNR after  : 9.0 dB
Gain       : +3.8 dB
```

**Generate all plots**
```bash
python visualizer.py
```

---

## Specifications

| Property | Value |
|---|---|
| Architecture | Weight-stationary 1D systolic array |
| Filter type | FIR — Low Pass, High Pass, Band Pass |
| Tap counts | 8, 16, 32 |
| Window functions | Hamming, Blackman |
| Sample rate | 1000 Hz |
| Default cutoff | 100 Hz |
| Verification error | < 2.22e-16 (machine precision) |
| Language | Python 3.10+ |
| Dependencies | NumPy, Matplotlib, Pillow |

---

## Built With

Python · NumPy · Matplotlib · Pillow
