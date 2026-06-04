from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .load_graph import ordered_nodes


def _save(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_graph(graph: nx.Graph, output: Path, seed: int = 7) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    pos = nx.spring_layout(graph, weight="weight", seed=seed, iterations=80)

    edges = sorted(
        graph.edges(data=True),
        key=lambda edge: float(edge[2].get("weight", 0.0)),
        reverse=True,
    )
    edges = edges[: min(1400, len(edges))]
    widths = [0.15 + 1.8 * float(data.get("weight", 0.0)) for _, _, data in edges]
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=[(u, v) for u, v, _ in edges],
        width=widths,
        alpha=0.16,
        edge_color="#606060",
        ax=ax,
    )

    nodes = ordered_nodes(graph)
    classes = [graph.nodes[node].get("class_label", "unknown") for node in nodes]
    unique_classes = sorted(set(classes), key=str)
    class_to_idx = {value: idx for idx, value in enumerate(unique_classes)}
    colors = [class_to_idx[value] for value in classes]
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=nodes,
        node_color=colors,
        cmap="tab20",
        node_size=38,
        linewidths=0.25,
        edgecolors="#1f1f1f",
        ax=ax,
    )
    ax.set_title("School contact network (aggregated weighted graph)")
    ax.set_axis_off()
    _save(fig, output)


def plot_sir_curve(mean_counts: pd.DataFrame, output: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(mean_counts["time"], mean_counts["susceptible"], label="S", color="#2f6fbb", lw=2)
    ax.plot(mean_counts["time"], mean_counts["infected"], label="I", color="#c43c39", lw=2)
    ax.plot(mean_counts["time"], mean_counts["recovered"], label="R", color="#3a8b4f", lw=2)
    if mean_counts["vaccinated"].max() > 0:
        ax.plot(mean_counts["time"], mean_counts["vaccinated"], label="V", color="#7a5195", lw=2)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean number of nodes")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, output)


def plot_beta_sweep(beta_summary: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(beta_summary))
    width = 0.36
    ax.bar(
        x - width / 2,
        beta_summary["final_infected_ratio"],
        width,
        label="Final infected ratio",
        color="#3b6ea8",
    )
    ax.bar(
        x + width / 2,
        beta_summary["peak_infected_count"] / beta_summary["nodes"],
        width,
        label="Peak infected ratio",
        color="#c75b39",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"beta={value:.3f}" for value in beta_summary["beta"]])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Ratio")
    ax.set_title("Outbreak sensitivity to infection rate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, output)


def plot_intervention_comparison(summary: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_summary = summary.sort_values("final_infected_ratio", ascending=False)
    colors = ["#c43c39" if name == "Baseline" else "#3b6ea8" for name in sorted_summary["scenario"]]
    ax.bar(sorted_summary["scenario"], sorted_summary["final_infected_ratio"], color=colors)
    ax.set_ylabel("Mean final infected ratio")
    ax.set_title("Intervention strategy comparison")
    ax.set_ylim(0, max(0.08, sorted_summary["final_infected_ratio"].max() * 1.15))
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)
    _save(fig, output)


def plot_diffusion_risk(
    graph: nx.Graph,
    risk_scores: pd.DataFrame,
    initial_infected: list[object],
    output: Path,
    seed: int = 7,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    pos = nx.spring_layout(graph, weight="weight", seed=seed, iterations=80)
    nodes = ordered_nodes(graph)
    risk_map = dict(zip(risk_scores["node"], risk_scores["diffusion_risk"]))
    colors = [risk_map.get(node, 0.0) for node in nodes]
    sizes = [95 if node in initial_infected else 38 for node in nodes]

    top_edges = sorted(
        graph.edges(data=True),
        key=lambda edge: float(edge[2].get("weight", 0.0)),
        reverse=True,
    )[: min(1400, graph.number_of_edges())]
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=[(u, v) for u, v, _ in top_edges],
        width=0.4,
        alpha=0.11,
        edge_color="#606060",
        ax=ax,
    )
    nodes_artist = nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=nodes,
        node_color=colors,
        cmap="magma",
        node_size=sizes,
        linewidths=[1.2 if node in initial_infected else 0.2 for node in nodes],
        edgecolors=["#111111" if node in initial_infected else "#333333" for node in nodes],
        ax=ax,
    )
    fig.colorbar(nodes_artist, ax=ax, fraction=0.035, pad=0.02, label="Diffusion risk")
    ax.set_title("Graph diffusion risk score from initial infected nodes")
    ax.set_axis_off()
    _save(fig, output)


def plot_risk_scatter(risk_scores: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        risk_scores["diffusion_risk"],
        risk_scores["infection_frequency"],
        s=22,
        alpha=0.72,
        color="#3b6ea8",
    )
    ax.set_xlabel("Diffusion risk score")
    ax.set_ylabel("Monte Carlo infection frequency")
    ax.set_title("Diffusion risk vs. simulated infection frequency")
    ax.grid(alpha=0.25)
    _save(fig, output)


def plot_smoothness(smoothness: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(smoothness["time"], smoothness["smoothness"], color="#3a8b4f", lw=2)
    ax.set_xlabel("Time step")
    ax.set_ylabel("x(t)^T L x(t)")
    ax.set_title("Graph signal smoothness of infection probability")
    ax.grid(alpha=0.25)
    _save(fig, output)

