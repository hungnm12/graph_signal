from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def simulate_sir(
    beta: float,
    gamma: float,
    steps: int,
    dt: float,
    s0: float,
    i0: float,
    r0: float,
) -> dict[str, np.ndarray]:
    s = np.zeros(steps + 1)
    i = np.zeros(steps + 1)
    r = np.zeros(steps + 1)
    s[0], i[0], r[0] = s0, i0, r0

    for t in range(steps):
        new_infections = beta * s[t] * i[t] * dt
        new_recoveries = gamma * i[t] * dt
        s[t + 1] = max(0.0, s[t] - new_infections)
        i[t + 1] = max(0.0, i[t] + new_infections - new_recoveries)
        r[t + 1] = min(1.0, r[t] + new_recoveries)

    return {"S": s, "I": i, "R": r}


def simulate_seir(
    beta: float,
    sigma: float,
    gamma: float,
    steps: int,
    dt: float,
    s0: float,
    e0: float,
    i0: float,
    r0: float,
) -> dict[str, np.ndarray]:
    s = np.zeros(steps + 1)
    e = np.zeros(steps + 1)
    i = np.zeros(steps + 1)
    r = np.zeros(steps + 1)
    s[0], e[0], i[0], r[0] = s0, e0, i0, r0

    for t in range(steps):
        new_exposed = beta * s[t] * i[t] * dt
        new_infectious = sigma * e[t] * dt
        new_recoveries = gamma * i[t] * dt
        s[t + 1] = max(0.0, s[t] - new_exposed)
        e[t + 1] = max(0.0, e[t] + new_exposed - new_infectious)
        i[t + 1] = max(0.0, i[t] + new_infectious - new_recoveries)
        r[t + 1] = min(1.0, r[t] + new_recoveries)

    return {"S": s, "E": e, "I": i, "R": r}


def summarize(curve: np.ndarray, dt: float) -> tuple[float, float]:
    peak_idx = int(np.argmax(curve))
    return float(curve[peak_idx]), float(peak_idx * dt)


def main() -> None:
    base_dir = Path(__file__).parent
    comparison_output = base_dir / "seir_vs_sir_curves.png"
    sir_output = base_dir / "sir_raw_curves.png"
    seir_output = base_dir / "seir_raw_curves.png"

    days = 80
    dt = 0.2
    steps = int(days / dt)

    sir = simulate_sir(
        beta=0.62,
        gamma=0.18,
        steps=steps,
        dt=dt,
        s0=0.99,
        i0=0.01,
        r0=0.0,
    )
    seir = simulate_seir(
        beta=0.62,
        sigma=0.24,
        gamma=0.18,
        steps=steps,
        dt=dt,
        s0=0.99,
        e0=0.0,
        i0=0.01,
        r0=0.0,
    )

    time = np.arange(steps + 1) * dt
    sir_peak, sir_peak_day = summarize(sir["I"], dt)
    seir_peak, seir_peak_day = summarize(seir["I"], dt)

    plt.rcParams.update(
        {
            "figure.facecolor": "#f7f3ea",
            "axes.facecolor": "#fffdf8",
            "axes.edgecolor": "#c9c0b3",
            "axes.labelcolor": "#202634",
            "xtick.color": "#4e596d",
            "ytick.color": "#4e596d",
            "font.size": 11,
        }
    )

    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.6])

    ax_main = fig.add_subplot(gs[0, :])
    ax_sir = fig.add_subplot(gs[1, 0])
    ax_seir = fig.add_subplot(gs[1, 1])

    ax_main.plot(time, sir["I"], color="#d95f02", linewidth=3, label="SIR infectious")
    ax_main.plot(time, seir["I"], color="#1b9e77", linewidth=3, label="SEIR infectious")
    ax_main.plot(
        time,
        seir["E"],
        color="#7570b3",
        linewidth=2.5,
        linestyle="--",
        label="SEIR exposed",
    )
    ax_main.set_title("Infectious dynamics: SIR peaks earlier, SEIR adds an incubation delay", fontsize=16, pad=12)
    ax_main.set_xlabel("Time (days)")
    ax_main.set_ylabel("Population fraction")
    ax_main.grid(alpha=0.22, linewidth=0.8)
    ax_main.legend(frameon=False, ncol=3, loc="upper right")
    ax_main.text(
        0.015,
        0.96,
        (
            f"SIR peak: {sir_peak:.2f} at day {sir_peak_day:.1f}\n"
            f"SEIR peak: {seir_peak:.2f} at day {seir_peak_day:.1f}"
        ),
        transform=ax_main.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        color="#202634",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f4efe5", "edgecolor": "#d6ccbd"},
    )

    for ax, model_name, series in [
        (ax_sir, "SIR state trajectories", sir),
        (ax_seir, "SEIR state trajectories", seir),
    ]:
        ax.plot(time, series["S"], color="#4daf4a", linewidth=2.4, label="S")
        if "E" in series:
            ax.plot(time, series["E"], color="#7570b3", linewidth=2.2, linestyle="--", label="E")
        ax.plot(time, series["I"], color="#d95f02", linewidth=2.4, label="I")
        ax.plot(time, series["R"], color="#377eb8", linewidth=2.4, label="R")
        ax.set_title(model_name, fontsize=13, pad=8)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Fraction")
        ax.grid(alpha=0.20, linewidth=0.8)
        ax.set_ylim(0.0, 1.02)
        ax.legend(frameon=False, ncol=4 if "E" in series else 3, loc="upper right")

    fig.suptitle("SEIR vs SIR Epidemic Curves", fontsize=22, fontweight="bold", color="#202634")
    fig.savefig(comparison_output, dpi=220)
    plt.close(fig)

    fig_sir, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(time, sir["S"], color="#4daf4a", linewidth=2.6, label="Susceptible")
    ax.plot(time, sir["I"], color="#d95f02", linewidth=2.8, label="Infectious")
    ax.plot(time, sir["R"], color="#377eb8", linewidth=2.6, label="Recovered")
    ax.set_title("Raw SIR Epidemic Curves", fontsize=18, pad=12)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Population fraction")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.22, linewidth=0.8)
    ax.legend(frameon=False)
    fig_sir.savefig(sir_output, dpi=220)
    plt.close(fig_sir)

    fig_seir, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(time, seir["S"], color="#4daf4a", linewidth=2.6, label="Susceptible")
    ax.plot(time, seir["E"], color="#7570b3", linewidth=2.6, linestyle="--", label="Exposed")
    ax.plot(time, seir["I"], color="#d95f02", linewidth=2.8, label="Infectious")
    ax.plot(time, seir["R"], color="#377eb8", linewidth=2.6, label="Recovered")
    ax.set_title("Raw SEIR Epidemic Curves", fontsize=18, pad=12)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Population fraction")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.22, linewidth=0.8)
    ax.legend(frameon=False)
    fig_seir.savefig(seir_output, dpi=220)
    plt.close(fig_seir)

    print(comparison_output)
    print(sir_output)
    print(seir_output)


if __name__ == "__main__":
    main()
