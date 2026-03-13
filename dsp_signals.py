

import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple
from systolic_array import SystolicArray, design_lowpass_fir, design_bandpass_fir


# ──────────────────────────────────────────────
# Signal Generators
# ──────────────────────────────────────────────
class SignalGenerator:
    """Factory for standard DSP test signals."""

    def __init__(self, fs: float = 1000.0, duration: float = 1.0):
        self.fs = fs
        self.duration = duration
        self.N = int(fs * duration)
        self.t = np.linspace(0, duration, self.N, endpoint=False)

    def sinusoid(self, freq: float, amplitude: float = 1.0,
                 phase: float = 0.0) -> np.ndarray:
        return amplitude * np.sin(2 * np.pi * freq * self.t + phase)

    def multi_tone(self, freqs: List[float],
                   amplitudes: List[float] = None) -> np.ndarray:
        if amplitudes is None:
            amplitudes = [1.0] * len(freqs)
        return sum(a * np.sin(2 * np.pi * f * self.t)
                   for f, a in zip(freqs, amplitudes))

    def noisy_sinusoid(self, freq: float, snr_db: float = 20.0) -> np.ndarray:
        clean = self.sinusoid(freq)
        signal_power = np.mean(clean ** 2)
        noise_power  = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), self.N)
        return clean + noise

    def chirp(self, f_start: float, f_end: float) -> np.ndarray:
        """Linear frequency sweep."""
        phase = 2 * np.pi * (f_start * self.t +
                             (f_end - f_start) / (2 * self.duration) * self.t ** 2)
        return np.sin(phase)

    def step(self) -> np.ndarray:
        sig = np.zeros(self.N)
        sig[self.N // 4:] = 1.0
        return sig

    def impulse(self) -> np.ndarray:
        sig = np.zeros(self.N)
        sig[self.N // 4] = 1.0
        return sig

    def ecg_like(self) -> np.ndarray:
        """Synthetic ECG-like signal (useful for bio-DSP demo)."""
        sig = np.zeros(self.N)
        period = int(self.fs * 0.8)  # 75 BPM
        for i in range(0, self.N - period, period):
            # P wave
            sig[i + int(period * 0.1):i + int(period * 0.2)] += \
                0.2 * np.hanning(int(period * 0.1))
            # QRS complex
            n_qrs = int(period * 0.05)
            sig[i + int(period * 0.3):i + int(period * 0.3) + n_qrs] += \
                1.0 * np.hanning(n_qrs)
            # T wave
            sig[i + int(period * 0.5):i + int(period * 0.65)] += \
                0.3 * np.hanning(int(period * 0.15))
        noise = np.random.normal(0, 0.05, self.N)
        return sig + noise

    def get_all_demo_signals(self) -> Dict[str, np.ndarray]:
        return {
            "50Hz Pure Tone":       self.sinusoid(50),
            "Multi-Tone (50+200+400Hz)": self.multi_tone([50, 200, 400],
                                                          [1.0, 0.7, 0.4]),
            "Noisy Sinusoid (SNR=15dB)": self.noisy_sinusoid(50, snr_db=15),
            "Chirp (10→450 Hz)":    self.chirp(10, 450),
            "Step Response":        self.step(),
            "ECG-like Signal":      self.ecg_like(),
        }


# ──────────────────────────────────────────────
# Filter Bank (multiple FIR designs to compare)
# ──────────────────────────────────────────────
@dataclass
class FilterSpec:
    name: str
    coefficients: np.ndarray
    description: str


class FilterBank:
    """Collection of FIR filter designs for demo."""

    def __init__(self, fs: float = 1000.0):
        self.fs = fs
        nyq = fs / 2

        self.filters: List[FilterSpec] = [
            FilterSpec(
                name="LPF-8tap (fc=100Hz)",
                coefficients=design_lowpass_fir(8,  100/nyq, "hamming"),
                description="8-tap Hamming windowed LPF, cutoff 100 Hz"
            ),
            FilterSpec(
                name="LPF-16tap (fc=100Hz)",
                coefficients=design_lowpass_fir(16, 100/nyq, "hamming"),
                description="16-tap Hamming windowed LPF, cutoff 100 Hz"
            ),
            FilterSpec(
                name="LPF-32tap (fc=100Hz)",
                coefficients=design_lowpass_fir(32, 100/nyq, "blackman"),
                description="32-tap Blackman windowed LPF — sharper roll-off"
            ),
            FilterSpec(
                name="BPF 80–150Hz (16tap)",
                coefficients=design_bandpass_fir(16, 80/nyq, 150/nyq),
                description="16-tap band-pass, 80–150 Hz"
            ),
            FilterSpec(
                name="HPF-16tap (fc=200Hz)",
                coefficients=(design_lowpass_fir(16, 200/nyq) * -1 +
                              np.eye(16)[8]),   # spectral inversion
                description="16-tap high-pass via spectral inversion"
            ),
        ]

    def get(self, name: str) -> FilterSpec:
        for f in self.filters:
            if f.name == name:
                return f
        raise KeyError(f"Filter '{name}' not found")


# ──────────────────────────────────────────────
# Benchmarking
# ──────────────────────────────────────────────
@dataclass
class BenchmarkResult:
    filter_name: str
    num_taps: int
    signal_length: int
    systolic_time_ms: float
    numpy_time_ms: float
    speedup: float          # numpy/systolic (educational comparison)
    max_error: float
    throughput_mops: float  # Million multiply-accumulates per second


class Benchmarker:
    """Times systolic vs numpy convolution, computes SNR improvements."""

    def run(self, filter_spec: FilterSpec,
            signal: np.ndarray, repeats: int = 5) -> BenchmarkResult:
        coeffs = filter_spec.coefficients
        arr = SystolicArray(coeffs)

        # Time systolic
        t0 = time.perf_counter()
        for _ in range(repeats):
            sa_out = arr.process_signal(signal)
            arr.reset()
        t1 = time.perf_counter()
        systolic_ms = (t1 - t0) / repeats * 1000

        # Time numpy
        t0 = time.perf_counter()
        for _ in range(repeats):
            np_out = np.convolve(signal, coeffs, mode='full')[:len(signal)]
        t1 = time.perf_counter()
        numpy_ms = (t1 - t0) / repeats * 1000

        max_err = float(np.max(np.abs(sa_out - np_out)))
        macs    = len(signal) * len(coeffs)
        throughput = macs / (systolic_ms * 1e-3) / 1e6

        return BenchmarkResult(
            filter_name=filter_spec.name,
            num_taps=len(coeffs),
            signal_length=len(signal),
            systolic_time_ms=systolic_ms,
            numpy_time_ms=numpy_ms,
            speedup=numpy_ms / systolic_ms if systolic_ms > 0 else 0,
            max_error=max_err,
            throughput_mops=throughput,
        )

    def run_all(self, filter_bank: FilterBank,
                signal: np.ndarray) -> List[BenchmarkResult]:
        results = []
        for fspec in filter_bank.filters:
            r = self.run(fspec, signal)
            results.append(r)
            print(f"  [{fspec.name}]  SA={r.systolic_time_ms:.2f}ms  "
                  f"NP={r.numpy_time_ms:.2f}ms  "
                  f"err={r.max_error:.1e}  "
                  f"MOPS={r.throughput_mops:.1f}")
        return results


# ──────────────────────────────────────────────
# Frequency Response
# ──────────────────────────────────────────────
def compute_frequency_response(coeffs: np.ndarray,
                                fs: float = 1000.0,
                                n_fft: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (freqs_hz, magnitude_db)."""
    H = np.fft.rfft(coeffs, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1/fs)
    mag_db = 20 * np.log10(np.abs(H) + 1e-12)
    return freqs, mag_db


def compute_snr(signal_clean: np.ndarray,
                signal_noisy: np.ndarray) -> float:
    noise = signal_noisy - signal_clean
    if np.mean(noise**2) < 1e-20:
        return float('inf')
    return 10 * np.log10(np.mean(signal_clean**2) / np.mean(noise**2))


def snr_improvement(original_noisy: np.ndarray,
                    filtered: np.ndarray,
                    reference_clean: np.ndarray) -> dict:
    snr_before = compute_snr(reference_clean, original_noisy)
    snr_after  = compute_snr(reference_clean[:len(filtered)], filtered)
    return {
        "snr_before_db": snr_before,
        "snr_after_db":  snr_after,
        "improvement_db": snr_after - snr_before,
    }


# ──────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  DSP Signal Layer — Self-Test")
    print("=" * 55)

    gen   = SignalGenerator(fs=1000, duration=1.0)
    bank  = FilterBank(fs=1000)
    bench = Benchmarker()

    signal = gen.noisy_sinusoid(50, snr_db=15)
    print(f"\n  Benchmarking all filters on {len(signal)}-sample noisy signal:")
    bench.run_all(bank, signal)

    # SNR test — add noise on top of clean, compensate for FIR group delay
    clean       = gen.sinusoid(50)
    sig_power   = np.mean(clean ** 2)
    noise_power = sig_power / (10 ** (5 / 10))   # 5 dB input SNR (low, to show improvement)
    np.random.seed(42)
    noise       = np.random.normal(0, np.sqrt(noise_power), len(clean))
    noisy       = clean + noise

    fspec    = bank.filters[1]   # 16-tap LPF
    arr      = SystolicArray(fspec.coefficients)
    filtered = arr.process_signal(noisy)

    # Linear-phase FIR has group delay = (N-1)/2 samples — shift to align
    delay    = (len(fspec.coefficients) - 1) // 2
    c        = clean[:-delay]
    f        = filtered[delay:]
    n_sig    = noisy[:-delay]

    snr_before = 10 * np.log10(np.mean(c**2) / np.mean((n_sig - c)**2))
    snr_after  = 10 * np.log10(np.mean(c**2) / np.mean((f   - c)**2))
    print(f"\n  SNR improvement with {fspec.name}:")
    print(f"    Before : {snr_before:.1f} dB")
    print(f"    After  : {snr_after:.1f} dB")
    print(f"    Gain   : +{snr_after - snr_before:.1f} dB")
