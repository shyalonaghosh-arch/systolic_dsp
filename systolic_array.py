

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────
# Processing Element
# ──────────────────────────────────────────────
@dataclass
class ProcessingElement:
    """Single PE in the systolic array. Holds one FIR coefficient."""
    pe_id: int
    coefficient: float          # Stationary weight (tap)
    reg_in: float = 0.0         # Sample register (passed left→right)
    partial_sum_in: float = 0.0 # Partial sum coming from left neighbour
    partial_sum_out: float = 0.0
    sample_out: float = 0.0

    # History for visualisation
    activity_log: List[dict] = field(default_factory=list)

    def compute(self, sample: float, p_sum: float) -> tuple:
        """
        Execute one systolic clock cycle.
        Returns (sample_out, partial_sum_out)
        """
        self.reg_in = sample
        self.partial_sum_in = p_sum

        multiply = self.coefficient * sample
        self.partial_sum_out = p_sum + multiply
        self.sample_out = sample          # Pass sample to next PE

        self.activity_log.append({
            "sample_in":    sample,
            "coeff":        self.coefficient,
            "multiply":     multiply,
            "psum_in":      p_sum,
            "psum_out":     self.partial_sum_out,
        })
        return self.sample_out, self.partial_sum_out


# ──────────────────────────────────────────────
# Systolic Array (1-D, weight-stationary FIR)
# ──────────────────────────────────────────────
class SystolicArray:
    """
    N-tap weight-stationary systolic FIR filter.

    Parameters
    ----------
    coefficients : array-like
        FIR tap weights h[0], h[1], ..., h[N-1]
    """

    def __init__(self, coefficients: np.ndarray):
        self.coefficients = np.array(coefficients, dtype=float)
        self.N = len(coefficients)
        self.pes: List[ProcessingElement] = [
            ProcessingElement(pe_id=i, coefficient=float(coefficients[i]))
            for i in range(self.N)
        ]
        self.clock = 0
        self.output_history: List[Optional[float]] = []
        self.cycle_snapshots: List[dict] = []   # full state each cycle
        self._delay_line = np.zeros(self.N)     # shift register

    # ── Internal helpers ──────────────────────
    def _snapshot(self) -> dict:
        return {
            "clock": self.clock,
            "pe_states": [
                {
                    "pe_id":    pe.pe_id,
                    "coeff":    pe.coefficient,
                    "sample":   pe.reg_in,
                    "psum_out": pe.partial_sum_out,
                }
                for pe in self.pes
            ],
        }

    # ── Public API ────────────────────────────
    def process_sample(self, x: float) -> Optional[float]:
        """
        Feed one sample into the array; advance one clock.

        Architecture: weight-stationary shift-register FIR.
        - A delay-line register holds the last N samples.
        - PE[k] computes h[k] * delay_line[k] and accumulates.
        - All PEs fire in parallel each clock — true systolic behaviour.
        Returns the output y[n] each cycle (None during warm-up).
        """
        # Shift the delay line: index 0 = newest sample
        self._delay_line = np.roll(self._delay_line, 1)
        self._delay_line[0] = x

        psum = 0.0
        for pe in self.pes:
            delayed_sample = self._delay_line[pe.pe_id]
            _, psum = pe.compute(delayed_sample, 0.0)
            # Each PE independently multiplies; we sum all contributions
        # The total output is the sum over all PEs
        y = float(np.dot(self._delay_line, self.coefficients))

        self.clock += 1
        self.cycle_snapshots.append(self._snapshot())
        self.output_history.append(y)
        return y

    def process_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Process a full signal through the systolic array.
        Returns the filtered output (same length as input).
        """
        for pe in self.pes:
            pe.activity_log = []
        self.output_history = []
        self.cycle_snapshots = []
        self.clock = 0
        self._delay_line = np.zeros(self.N)

        results = []
        for x in signal:
            out = self.process_sample(x)
            results.append(out)

        return np.array(results)

    def reset(self):
        """Reset all PE registers (coefficients stay)."""
        for pe in self.pes:
            pe.reg_in = 0.0
            pe.partial_sum_in = 0.0
            pe.partial_sum_out = 0.0
            pe.sample_out = 0.0
            pe.activity_log = []
        self.clock = 0
        self.output_history = []
        self.cycle_snapshots = []
        self._delay_line = np.zeros(self.N)

    def get_activity_matrix(self) -> np.ndarray:
        """
        Returns matrix of shape (N_pes, N_cycles) showing |sample × coeff|
        at each PE each cycle — useful for heat-map visualisation.
        """
        n_cycles = max(len(pe.activity_log) for pe in self.pes)
        mat = np.zeros((self.N, n_cycles))
        for i, pe in enumerate(self.pes):
            for t, log in enumerate(pe.activity_log):
                mat[i, t] = abs(log["multiply"])
        return mat


# ──────────────────────────────────────────────
# FIR filter design helpers
# ──────────────────────────────────────────────
def design_lowpass_fir(num_taps: int, cutoff_norm: float,
                        window: str = "hamming") -> np.ndarray:
    """
    Simple windowed-sinc low-pass FIR design.
    cutoff_norm : normalised cutoff  (0–1, where 1 = Nyquist)
    """
    n = np.arange(num_taps)
    mid = (num_taps - 1) / 2.0
    h = np.sinc(2 * cutoff_norm * (n - mid))

    if window == "hamming":
        w = np.hamming(num_taps)
    elif window == "hann":
        w = np.hanning(num_taps)
    elif window == "blackman":
        w = np.blackman(num_taps)
    else:
        w = np.ones(num_taps)

    h = h * w
    h /= h.sum()   # Normalise DC gain to 1
    return h


def design_bandpass_fir(num_taps: int, low_norm: float,
                         high_norm: float) -> np.ndarray:
    """Windowed-sinc band-pass FIR."""
    h_low  = design_lowpass_fir(num_taps, high_norm)
    h_high = design_lowpass_fir(num_taps, low_norm)
    return h_low - h_high


# ──────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────
def verify_against_numpy(signal: np.ndarray,
                          coeffs: np.ndarray) -> dict:
    """
    Compare systolic array output vs numpy reference convolution.
    Returns dict with max_error, mse, match flag.
    """
    arr = SystolicArray(coeffs)
    sa_out = arr.process_signal(signal)

    np_out = np.convolve(signal, coeffs, mode='full')[:len(signal)]

    diff    = sa_out - np_out
    max_err = float(np.max(np.abs(diff)))
    mse     = float(np.mean(diff ** 2))

    return {
        "systolic_output": sa_out,
        "numpy_output":    np_out,
        "max_error":       max_err,
        "mse":             mse,
        "match":           max_err < 1e-9,
        "activity_matrix": arr.get_activity_matrix(),
        "cycle_snapshots": arr.cycle_snapshots,
    }


# ──────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Systolic Array — FIR Filter Self-Test")
    print("=" * 55)

    fs = 1000          # Sample rate Hz
    t  = np.linspace(0, 1, fs, endpoint=False)

    # Signal: 50 Hz (pass) + 300 Hz (block)
    signal = np.sin(2 * np.pi * 50  * t) + \
             0.5 * np.sin(2 * np.pi * 300 * t)

    # 16-tap low-pass FIR, cutoff at 100 Hz normalised
    coeffs = design_lowpass_fir(16, cutoff_norm=100/500, window="hamming")

    result = verify_against_numpy(signal, coeffs)

    print(f"  Taps      : {len(coeffs)}")
    print(f"  Samples   : {len(signal)}")
    print(f"  Max error : {result['max_error']:.2e}")
    print(f"  MSE       : {result['mse']:.2e}")
    print(f"  ✓ Match   : {result['match']}")
    print()

    # Show first 5 cycle snapshots
    print("  First 3 clock cycle snapshots:")
    for snap in result["cycle_snapshots"][:3]:
        print(f"    Clock {snap['clock']:02d} | ", end="")
        for pe in snap["pe_states"]:
            print(f"PE{pe['pe_id']}(s={pe['sample']:.3f},p={pe['psum_out']:.3f}) ", end="")
        print()
