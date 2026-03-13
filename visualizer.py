

import os
OUT = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import warnings
warnings.filterwarnings("ignore")

from systolic_array import SystolicArray, design_lowpass_fir
from dsp_signals import (SignalGenerator, FilterBank, Benchmarker,
                          compute_frequency_response, snr_improvement)


# ──────────────────────────────────────────────
# Colour Palette  (dark industrial theme)
# ──────────────────────────────────────────────
BG      = "#0d0f14"
PANEL   = "#13161e"
ACCENT1 = "#00e5ff"   # cyan
ACCENT2 = "#ff4081"   # pink
ACCENT3 = "#69ff47"   # green
ACCENT4 = "#ffd740"   # amber
MUTED   = "#4a5568"
TEXT    = "#e2e8f0"

def style():
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    PANEL,
        "axes.edgecolor":    MUTED,
        "axes.labelcolor":   TEXT,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "text.color":        TEXT,
        "grid.color":        "#1e2533",
        "grid.linewidth":    0.6,
        "lines.linewidth":   1.6,
        "font.family":       "monospace",
    })

# ──────────────────────────────────────────────
# 1.  Main 6-panel static dashboard
# ──────────────────────────────────────────────
def plot_main_dashboard():
    style()
    fs = 1000
    gen  = SignalGenerator(fs=fs, duration=1.0)
    bank = FilterBank(fs=fs)

    # Signals
    clean   = gen.sinusoid(50)
    noisy   = gen.noisy_sinusoid(50, snr_db=12)
    multi   = gen.multi_tone([50, 200, 400], [1.0, 0.6, 0.35])
    chirp   = gen.chirp(10, 450)

    # Filter — 16-tap LPF
    fspec  = bank.filters[1]
    arr    = SystolicArray(fspec.coefficients)
    filt_n = arr.process_signal(noisy)
    arr.reset()
    filt_m = arr.process_signal(multi)

    snr = snr_improvement(noisy, filt_n, clean)
    freqs, mag_db = compute_frequency_response(fspec.coefficients, fs)

    t   = gen.t
    fig = plt.figure(figsize=(18, 11), facecolor=BG)
    fig.suptitle("SYSTOLIC ARRAY  ·  FIR FILTER SIMULATION  ·  DSP FRAME",
                 fontsize=14, color=ACCENT1, fontweight="bold", y=0.98)

    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.52, wspace=0.38,
                            left=0.06, right=0.97,
                            top=0.93, bottom=0.07)

    # ── Panel A: Input vs Output (noisy) ──────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(t[:300], noisy[:300],  color=ACCENT2, alpha=0.55, lw=0.9,
             label="Noisy input")
    ax1.plot(t[:300], filt_n[:300], color=ACCENT3, lw=1.8,
             label="Systolic FIR output")
    ax1.set_title("A  ·  Noise Suppression  (50 Hz + AWGN)", color=ACCENT1)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(framealpha=0.2, loc="upper right")
    ax1.grid(True)
    ax1.text(0.02, 0.88,
             f"SNR before: {snr['snr_before_db']:.1f} dB  →  after: {snr['snr_after_db']:.1f} dB  (+{snr['improvement_db']:.1f} dB)",
             transform=ax1.transAxes, color=ACCENT4, fontsize=8)

    # ── Panel B: Frequency Response ───────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(freqs, mag_db, color=ACCENT1, lw=1.8)
    ax2.axvline(100, color=ACCENT2, lw=1.2, ls="--", alpha=0.7,
                label="fc = 100 Hz")
    ax2.set_title("B  ·  Filter Freq. Response", color=ACCENT1)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.set_ylim(-80, 5)
    ax2.legend(framealpha=0.2)
    ax2.grid(True)

    # ── Panel C: Multi-tone separation ────────
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(t[:400], multi[:400],  color=ACCENT2, alpha=0.5, lw=0.9,
             label="Multi-tone (50+200+400 Hz)")
    ax3.plot(t[:400], filt_m[:400], color=ACCENT3, lw=1.8,
             label="After LPF (keeps 50 Hz)")
    ax3.set_title("C  ·  Multi-Tone Separation", color=ACCENT1)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Amplitude")
    ax3.legend(framealpha=0.2)
    ax3.grid(True)

    # ── Panel D: Systolic activity heatmap ────
    ax4 = fig.add_subplot(gs[1, 2])
    arr2 = SystolicArray(fspec.coefficients)
    arr2.process_signal(chirp)
    mat  = arr2.get_activity_matrix()
    cmap = LinearSegmentedColormap.from_list(
               "sa_heat", [PANEL, ACCENT1, ACCENT4, ACCENT2])
    im = ax4.imshow(mat[:, :80], aspect="auto", cmap=cmap, origin="lower")
    ax4.set_title("D  ·  PE Activity Heatmap", color=ACCENT1)
    ax4.set_xlabel("Clock Cycle")
    ax4.set_ylabel("PE index (tap)")
    plt.colorbar(im, ax=ax4, label="|coeff × sample|")

    # ── Panel E: Tap weight coefficients ──────
    ax5 = fig.add_subplot(gs[2, 0])
    taps = np.arange(len(fspec.coefficients))
    markerline, stemlines, baseline = ax5.stem(taps, fspec.coefficients)
    plt.setp(stemlines, color=ACCENT1, linewidth=1.5)
    plt.setp(markerline, color=ACCENT1, markersize=6)
    plt.setp(baseline, color=MUTED, linewidth=1.0)
    ax5.set_title("E  ·  FIR Tap Coefficients", color=ACCENT1)
    ax5.set_xlabel("Tap index")
    ax5.set_ylabel("Weight")
    ax5.grid(True)

    # ── Panel F: Benchmark bar chart ──────────
    ax6 = fig.add_subplot(gs[2, 1:])
    bench   = Benchmarker()
    results = bench.run_all(bank, noisy)
    names   = [r.filter_name.split(" ")[0] for r in results]
    sa_t    = [r.systolic_time_ms  for r in results]
    np_t    = [r.numpy_time_ms     for r in results]

    x  = np.arange(len(names))
    w  = 0.35
    ax6.bar(x - w/2, sa_t, w, color=ACCENT1,  label="Systolic Array",  alpha=0.85)
    ax6.bar(x + w/2, np_t, w, color=ACCENT4, label="NumPy convolve", alpha=0.85)
    ax6.set_title("F  ·  Execution Time Benchmark", color=ACCENT1)
    ax6.set_xticks(x)
    ax6.set_xticklabels(names, rotation=18, ha="right", fontsize=7)
    ax6.set_ylabel("Time (ms)")
    ax6.legend(framealpha=0.2)
    ax6.grid(True, axis="y")

    plt.savefig(os.path.join(OUT, "dashboard.png"),
                dpi=150, bbox_inches="tight", facecolor=BG)
    print("  ✓ Saved  dashboard.png")
    plt.show()


# ──────────────────────────────────────────────
# 2.  Systolic Array Step Animation
# ──────────────────────────────────────────────
def animate_systolic_array(n_taps: int = 8, n_steps: int = 24,
                            save_gif: bool = True):
    """
    Animate samples flowing through PE cells clock-by-clock.
    Green = active multiply, blue = data register, grey = idle.
    """
    style()
    fs     = 1000
    gen    = SignalGenerator(fs=fs, duration=0.05)
    coeffs = design_lowpass_fir(n_taps, 0.2, "hamming")
    arr    = SystolicArray(coeffs)

    signal = gen.multi_tone([50, 200], [1.0, 0.5])
    padded = np.concatenate([signal, np.zeros(n_taps - 1)])
    samples_to_show = padded[:n_steps]

    # Pre-run to capture snapshots
    arr.process_signal(signal)
    snaps = arr.cycle_snapshots[:n_steps]

    fig, ax = plt.subplots(figsize=(14, 5), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.set_xlim(-0.5, n_taps - 0.5)
    ax.set_ylim(-0.5, 2.5)
    ax.axis("off")
    title_txt = ax.set_title("SYSTOLIC ARRAY  —  Clock 0",
                              color=ACCENT1, fontsize=13, pad=12)

    cell_w, cell_h = 0.85, 1.2
    boxes, coeff_txts, sample_txts, psum_txts = [], [], [], []

    for i in range(n_taps):
        rect = mpatches.FancyBboxPatch(
            (i - cell_w/2, 0.5), cell_w, cell_h,
            boxstyle="round,pad=0.06",
            linewidth=1.5, edgecolor=ACCENT1,
            facecolor="#1a2030"
        )
        ax.add_patch(rect)
        boxes.append(rect)

        ax.text(i, 2.0, f"PE{i}", ha="center", va="center",
                color=ACCENT1, fontsize=8, fontweight="bold")

        ct = ax.text(i, 1.7, f"h={coeffs[i]:.3f}",
                     ha="center", va="center", color=ACCENT4, fontsize=7.5)
        coeff_txts.append(ct)

        st = ax.text(i, 1.15, "x=—",
                     ha="center", va="center", color=TEXT, fontsize=8)
        sample_txts.append(st)

        pt = ax.text(i, 0.72, "p=—",
                     ha="center", va="center", color=ACCENT3, fontsize=7.5)
        psum_txts.append(pt)

        # Arrow between PEs
        if i < n_taps - 1:
            ax.annotate("", xy=(i + 0.48, 1.1), xytext=(i + 0.52 + (1 - cell_w)/2 - 0.04, 1.1),
                        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))

    clock_txt = ax.text(0.5, 0.02, "Sample in: —",
                        transform=ax.transAxes, ha="center",
                        color=ACCENT2, fontsize=9)

    def update(frame):
        if frame >= len(snaps):
            return
        snap = snaps[frame]
        title_txt.set_text(f"SYSTOLIC ARRAY  —  Clock {snap['clock']:02d}")

        for pe_state in snap["pe_states"]:
            i  = pe_state["pe_id"]
            s  = pe_state["sample"]
            ps = pe_state["psum_out"]
            active = abs(s) > 0.001

            boxes[i].set_facecolor("#0d2030" if active else "#1a2030")
            boxes[i].set_edgecolor(ACCENT3 if active else ACCENT1)
            sample_txts[i].set_text(f"x={s:.3f}")
            psum_txts[i].set_text(f"p={ps:.3f}")
            sample_txts[i].set_color(ACCENT3 if active else TEXT)

        x_in = samples_to_show[frame] if frame < len(samples_to_show) else 0.0
        clock_txt.set_text(f"Input sample: {x_in:.4f}")

    ani = animation.FuncAnimation(fig, update, frames=len(snaps),
                                   interval=420, blit=False, repeat=True)

    if save_gif:
        try:
            ani.save(os.path.join(OUT, "systolic_animation.gif"),
                     writer="pillow", fps=2.5, dpi=110)
            print("  ✓ Saved  systolic_animation.gif")
        except Exception as e:
            print(f"  ✗ GIF save failed ({e}) — showing interactively")

    plt.tight_layout()
    plt.show()
    return ani


# ──────────────────────────────────────────────
# 3.  Frequency Domain Comparison (all filters)
# ──────────────────────────────────────────────
def plot_filter_bank_response():
    style()
    fs   = 1000
    bank = FilterBank(fs=fs)
    colors = [ACCENT1, ACCENT2, ACCENT3, ACCENT4, "#c084fc"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    fig.suptitle("Filter Bank — Frequency & Phase Responses",
                 color=ACCENT1, fontsize=13)

    for ax in axes:
        ax.set_facecolor(PANEL)

    for fspec, col in zip(bank.filters, colors):
        freqs, mag = compute_frequency_response(fspec.coefficients, fs)
        H = np.fft.rfft(fspec.coefficients, n=1024)

        axes[0].plot(freqs, mag, color=col, lw=1.7, label=fspec.name)
        axes[1].plot(freqs, np.unwrap(np.angle(H)), color=col,
                     lw=1.5, label=fspec.name)

    axes[0].set_title("Magnitude Response (dB)", color=ACCENT1)
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("dB")
    axes[0].set_ylim(-90, 5)
    axes[0].axhline(-3,  color=MUTED, ls=":", lw=1)
    axes[0].axhline(-40, color=MUTED, ls=":", lw=1)
    axes[0].legend(fontsize=7.5, framealpha=0.15)
    axes[0].grid(True)

    axes[1].set_title("Phase Response (rad)", color=ACCENT1)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Phase (rad)")
    axes[1].legend(fontsize=7.5, framealpha=0.15)
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "filter_bank_response.png"),
                dpi=150, bbox_inches="tight", facecolor=BG)
    print("  ✓ Saved  filter_bank_response.png")
    plt.show()


# ──────────────────────────────────────────────
# 4.  ECG Denoising Demo  (highlight use-case)
# ──────────────────────────────────────────────
def plot_ecg_demo():
    style()
    fs  = 1000
    gen = SignalGenerator(fs=fs, duration=3.0)
    ecg = gen.ecg_like()

    coeffs   = design_lowpass_fir(32, 80/500, "blackman")
    arr      = SystolicArray(coeffs)
    filtered = arr.process_signal(ecg)

    t = gen.t
    fig, axes = plt.subplots(2, 1, figsize=(14, 7),
                              facecolor=BG, sharex=True)
    fig.suptitle("ECG Denoising via Systolic FIR  (32-tap Blackman LPF)",
                 color=ACCENT1, fontsize=13)

    axes[0].plot(t, ecg,      color=ACCENT2, lw=0.9, alpha=0.8,
                 label="Raw (with noise)")
    axes[0].set_title("Input ECG Signal", color=ACCENT1)
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(framealpha=0.2)
    axes[0].grid(True)

    axes[1].plot(t[:len(filtered)], filtered,
                 color=ACCENT3, lw=1.6, label="Filtered by systolic FIR")
    axes[1].set_title("Filtered Output", color=ACCENT1)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend(framealpha=0.2)
    axes[1].grid(True)

    for ax in axes:
        ax.set_facecolor(PANEL)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "ecg_demo.png"),
                dpi=150, bbox_inches="tight", facecolor=BG)
    print("  ✓ Saved  ecg_demo.png")
    plt.show()


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Systolic Array Visualizer — Generating all plots")
    print("=" * 55)

    print("\n[1/4] Main 6-panel dashboard...")
    plot_main_dashboard()

    print("\n[2/4] Systolic array animation...")
    animate_systolic_array(n_taps=8, n_steps=20, save_gif=True)

    print("\n[3/4] Filter bank frequency response...")
    plot_filter_bank_response()

    print("\n[4/4] ECG denoising demo...")
    plot_ecg_demo()

    print("\n  ✓ All outputs saved to /mnt/user-data/outputs/")
