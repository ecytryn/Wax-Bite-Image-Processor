"""
period_phase_asym.py

Computes and plots eruption interval, mean eruption period, and phase
asymmetry from binary tooth data.  Python equivalent of the MATLAB
PlotPeriodPhaseAsym pipeline (Excel2mat → PlotLGData → PlotPeriodPhaseAsym).

Usage
-----
Called from format_plot.py during the 'analyze' step, or standalone:

    from period_phase_asym import plot_period_phase_asym
    plot_period_phase_asym("processed/output/binary data.csv",
                           "processed/output/",
                           include_mean=True)
"""

import os
from collections import defaultdict
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# ── data extraction ────────────────────────────────────────────────


def extract_eruption_events(binary_csv_path: str) -> np.ndarray:
    """
    Read binary data CSV and extract eruption events via diff (0→1
    transitions), mirroring the MATLAB Excel2mat.m logic.

    Returns
    -------
    Array of shape (N, 2) with columns [tooth_position, eruption_day]
    where eruption_day is integer days since the first observation.
    Sorted by (position, day).
    """
    df = pd.read_csv(binary_csv_path)

    # parse dates, compute integer day offsets
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in df["date"]]
    day_numbers = [(d - dates[0]).days for d in dates]

    # tooth-position columns (skip pandas index and 'date')
    position_cols = [c for c in df.columns if c not in ("Unnamed: 0", "date")]

    events = []
    for col in position_cols:
        position = int(col)
        values = df[col].to_numpy()

        last_known = values[0]
        for i in range(1, len(values)):
            curr_val = values[i]
            if np.isnan(curr_val):
                continue
            if not np.isnan(last_known):
                if int(last_known) == 0 and int(curr_val) == 1:
                    events.append([position, day_numbers[i]])
            last_known = curr_val

    events = np.array(events) if events else np.empty((0, 2))

    if len(events) > 0:
        order = np.lexsort((events[:, 1], events[:, 0]))
        events = events[order]

    return events


# ── eruption interval ──────────────────────────────────────────────


def compute_lapsed_time(events: np.ndarray) -> np.ndarray:
    """
    Days since the previous eruption at the **same** tooth position.
    First eruption at each position is marked -1 (undefined).
    """
    lapsed = np.full(len(events), -1.0)
    for i in range(1, len(events)):
        if events[i, 0] == events[i - 1, 0]:  # same position
            lapsed[i] = events[i, 1] - events[i - 1, 1]
    return lapsed


# ── phase asymmetry ────────────────────────────────────────────────


def compute_phase_asymmetry(events: np.ndarray):
    """
    For each eruption, compute its phase relative to the cycles of the
    left (position-1) and right (position+1) neighbors, then return

        avg_phase  = cos(π (φ_r + φ_l))
        asym_phase = sin(π (φ_r − φ_l)) / π

    Both arrays have length len(events); entries are NaN where a
    neighbor lacks bracketing eruptions.
    """
    n = len(events)
    avg_phase = np.full(n, np.nan)
    asym_phase = np.full(n, np.nan)

    # lookup: position → sorted eruption times
    pos_times: dict[int, list[float]] = defaultdict(list)
    for i in range(n):
        pos_times[int(events[i, 0])].append(events[i, 1])
    for pos in pos_times:
        pos_times[pos].sort()

    for k in range(n):
        pos = int(events[k, 0])
        t = events[k, 1]

        left_phase = _neighbor_phase(t, pos_times.get(pos - 1, []))
        right_phase = _neighbor_phase(t, pos_times.get(pos + 1, []))

        if left_phase is not None and right_phase is not None:
            avg_phase[k] = np.cos(np.pi * (right_phase + left_phase))
            asym_phase[k] = np.sin(np.pi * (right_phase - left_phase)) / np.pi

    return avg_phase, asym_phase


def _neighbor_phase(t: float, neighbor_times: list[float]):
    """
    Where does time *t* fall in the eruption cycle of a neighbouring
    position?  Returns a value in [0, 1) or None.
    """
    if len(neighbor_times) < 2:
        return None
    prev_times = [nt for nt in neighbor_times if nt < t]
    next_times = [nt for nt in neighbor_times if nt >= t]
    if not prev_times or not next_times:
        return None
    prev = max(prev_times)
    nxt = min(next_times)
    if nxt == prev:
        return None
    return (t - prev) / (nxt - prev)


# ── colormaps (match MATLAB) ──────────────────────────────────────


def _jet_lower_cmap():
    """Lower half of MATLAB's jet (blue → cyan → green → yellow)."""
    jet = plt.cm.jet
    colors = jet(np.linspace(0, 0.5, 256))
    return LinearSegmentedColormap.from_list("jet_lower", colors)


def _pwg_sqrt_cmap():
    """
    Purple-white-green diverging colormap with element-wise sqrt,
    matching MATLAB's sqrt(customcolormap_preset('purple-white-green')).
    """
    purple = np.array([0.4, 0.0, 0.6])
    white = np.array([1.0, 1.0, 1.0])
    green = np.array([0.0, 0.5, 0.0])

    n = 128
    upper = np.array([purple + (white - purple) * i / (n - 1) for i in range(n)])
    lower = np.array([white + (green - white) * i / (n - 1) for i in range(n)])
    colors = np.sqrt(np.clip(np.vstack([upper, lower]), 0, 1))

    return LinearSegmentedColormap.from_list("pwg_sqrt", colors)


# ── main plotting function ────────────────────────────────────────


def plot_period_phase_asym(
    binary_csv_path: str,
    output_path: str,
    include_mean: bool = True,
) -> plt.Figure:
    """
    Produce a 2- or 3-panel figure and save to *output_path*.

    Panel A  – eruption interval (scatter, colored by days since last eruption)
    Panel B  – mean eruption interval ± std per position (optional)
    Panel C  – asymmetry of relative phase

    Parameters
    ----------
    binary_csv_path : path to ``binary data.csv``
    output_path     : directory in which to save the PNG
    include_mean    : if True, include the mean-period middle panel

    Returns
    -------
    The matplotlib Figure.
    """
    # ── font setup (serif to match MATLAB) ────────────────────────
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]
    mpl.rcParams["mathtext.fontset"] = "cm"  # Computer Modern for math

    # ── data ──────────────────────────────────────────────────────
    events = extract_eruption_events(binary_csv_path)
    if len(events) == 0:
        raise RuntimeError("No eruption events found in binary data.")

    lapsed = compute_lapsed_time(events)
    _, asym_phase = compute_phase_asymmetry(events)

    defined = lapsed >= 0
    undefined = ~defined

    # mean ± std per position
    unique_positions = sorted(set(events[defined, 0].astype(int)))
    mean_period, std_period = [], []
    for pos in unique_positions:
        vals = lapsed[defined & (events[:, 0] == pos)]
        mean_period.append(np.mean(vals) if len(vals) > 0 else np.nan)
        std_period.append(np.std(vals) if len(vals) > 0 else np.nan)
    mean_period = np.array(mean_period)
    std_period = np.array(std_period)
    global_mean = np.mean(lapsed[defined])

    # axis limits
    end_day = np.max(events[:, 1])
    jaw_end_l = min(events[:, 0]) - 2
    jaw_end_r = max(events[:, 0]) + 2

    # ── figure layout ─────────────────────────────────────────────
    # All panels share the same plot width; colorbars are placed
    # explicitly so the mean panel (no colorbar) aligns properly.
    FS1 = 20  # panel letters
    FS2 = 14  # titles / axis labels
    FS3 = 12  # colorbar labels
    dot_size = 50

    plot_left = 0.10
    plot_width = 0.62
    cbar_gap = 0.02
    cbar_left = plot_left + plot_width + cbar_gap
    cbar_width = 0.02

    fig = plt.figure(figsize=(10, 8))

    if include_mean:
        ax1 = fig.add_axes([plot_left, 0.65, plot_width, 0.23])
        cax1 = fig.add_axes([cbar_left, 0.65, cbar_width, 0.23])
        ax2 = fig.add_axes([plot_left, 0.42, plot_width, 0.13])
        ax3 = fig.add_axes([plot_left, 0.10, plot_width, 0.23])
        cax3 = fig.add_axes([cbar_left, 0.10, cbar_width, 0.23])
    else:
        ax1 = fig.add_axes([plot_left, 0.55, plot_width, 0.35])
        cax1 = fig.add_axes([cbar_left, 0.55, cbar_width, 0.35])
        ax3 = fig.add_axes([plot_left, 0.10, plot_width, 0.35])
        cax3 = fig.add_axes([cbar_left, 0.10, cbar_width, 0.35])
        ax2 = None

    letter_y = end_day * 1.2  # y-position for panel letters

    # ── Panel A: eruption interval ────────────────────────────────
    cmap_a = _jet_lower_cmap()

    sc1 = ax1.scatter(
        events[defined, 0],
        events[defined, 1],
        s=dot_size,
        c=lapsed[defined],
        cmap=cmap_a,
        vmin=0,
        vmax=65,
        edgecolors="none",
    )
    # undefined points: open circles (matching MATLAB)
    ax1.scatter(
        events[undefined, 0],
        events[undefined, 1],
        s=dot_size,
        facecolors="none",
        edgecolors=(0.6, 0.6, 0.6),
        linewidths=0.8,
    )

    cb1 = fig.colorbar(sc1, cax=cax1)
    cb1.set_label("Days since last eruption", fontsize=FS3)
    ax1.set_ylabel("Day of eruption", fontsize=FS2)
    ax1.set_title("Eruption interval", fontsize=FS2)
    ax1.set_xlim(jaw_end_l, jaw_end_r)
    ax1.set_ylim(-15, end_day + 15)
    ax1.tick_params(labelsize=FS3)
    cax1.tick_params(labelsize=FS3)
    ax1.text(jaw_end_l - 12, letter_y, "A", fontsize=FS1, fontweight="bold")

    # ── Panel B: mean eruption interval ───────────────────────────
    if ax2 is not None:
        ax2.axhline(global_mean, color="k", linestyle=":", linewidth=0.8)
        ax2.errorbar(
            unique_positions,
            mean_period,
            yerr=std_period,
            fmt="-o",
            markersize=3,
            capsize=2,
            color="tab:orange",
            ecolor="tab:orange",
        )
        ax2.set_title("Mean eruption interval", fontsize=FS2)
        ax2.set_xlim(jaw_end_l, jaw_end_r)
        y_max = np.nanmax(mean_period + std_period) * 1.2
        y_max = min(y_max, 100) if not np.isnan(y_max) else 60
        ax2.set_ylim(0, y_max)
        ax2.tick_params(labelsize=FS3)

    # ── Panel C: phase asymmetry ──────────────────────────────────
    cmap_c = _pwg_sqrt_cmap()

    not_nan = ~np.isnan(asym_phase)
    is_nan = np.isnan(asym_phase)

    edge_color = (0.8, 0.8, 0.8, 1) if end_day < 500 else "none"
    sc3 = ax3.scatter(
        events[not_nan, 0],
        events[not_nan, 1],
        s=dot_size,
        c=asym_phase[not_nan],
        cmap=cmap_c,
        vmin=-0.2,
        vmax=0.2,
        edgecolors=edge_color,
    )
    # NaN points: open circles
    ax3.scatter(
        events[is_nan, 0],
        events[is_nan, 1],
        s=dot_size,
        facecolors="none",
        edgecolors=(0.6, 0.6, 0.6),
        linewidths=0.8,
    )

    cb3 = fig.colorbar(sc3, cax=cax3)
    cb3.set_label(
        r"$\sin(\pi(\phi_r - \phi_l))/\pi$",
        fontsize=FS3 * 1.3,
    )
    cb3.set_ticks([-0.15, 0, 0.15])
    ax3.set_xlabel("Tooth Position", fontsize=FS2)
    ax3.set_ylabel("Day of eruption", fontsize=FS2)
    ax3.set_title("Asymmetry of relative phase", fontsize=FS2)
    ax3.set_xlim(jaw_end_l, jaw_end_r)
    ax3.set_ylim(-15, end_day + 15)
    ax3.tick_params(labelsize=FS3)
    cax3.tick_params(labelsize=FS3)
    ax3.text(jaw_end_l - 12, letter_y, "B", fontsize=FS1, fontweight="bold")

    # ── save ──────────────────────────────────────────────────────
    fig_path = os.path.join(output_path, "period_phase_asym.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Saved {fig_path}")

    return fig
