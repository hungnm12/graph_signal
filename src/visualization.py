from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

from .load_graph import ordered_nodes


def _save(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_animation(animation: FuncAnimation, output: Path, fps: int = 8) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=fps), dpi=120)
    plt.close(animation._fig)


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


def plot_risk_scatter_3d(
    risk_scores: pd.DataFrame,
    output: Path,
    annotate_top_k: int = 8,
    azim: float = -61,
) -> None:
    ranked = risk_scores.sort_values("diffusion_risk", ascending=False).reset_index(drop=True)
    x = ranked.index.to_numpy()
    y = ranked["diffusion_risk"].to_numpy()
    z = ranked["infection_frequency"].to_numpy()
    colors = ranked["seed_infected"].to_numpy()

    fig = plt.figure(figsize=(10.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        x,
        y,
        z,
        c=colors,
        cmap="coolwarm",
        s=48,
        alpha=0.92,
        depthshade=True,
        edgecolors="#101014",
        linewidths=0.45,
    )
    for idx, row in ranked.head(annotate_top_k).iterrows():
        ax.text(
            idx,
            float(row["diffusion_risk"]) + 0.01,
            float(row["infection_frequency"]) + 0.012,
            str(int(row["node"])),
            fontsize=8,
            color="#121212",
        )

    ax.set_title("3D Risk Landscape from Graph Signal Results", pad=18, fontsize=15)
    ax.set_xlabel("Node rank by diffusion risk", labelpad=12)
    ax.set_ylabel("Diffusion risk score", labelpad=12)
    ax.set_zlabel("Infection frequency", labelpad=10)
    ax.view_init(elev=26, azim=azim)
    ax.xaxis.pane.set_facecolor((0.95, 0.97, 1.0, 0.88))
    ax.yaxis.pane.set_facecolor((0.98, 0.95, 0.93, 0.84))
    ax.zaxis.pane.set_facecolor((0.95, 0.98, 0.95, 0.84))
    ax.grid(True, alpha=0.2)

    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.72, pad=0.08)
    colorbar.set_label("Initial infected seed", rotation=270, labelpad=18)
    _save(fig, output)


def animate_risk_scatter_3d_rotation(
    risk_scores: pd.DataFrame,
    output: Path,
    frames: int = 48,
    fps: int = 12,
) -> None:
    ranked = risk_scores.sort_values("diffusion_risk", ascending=False).reset_index(drop=True)
    x = ranked.index.to_numpy()
    y = ranked["diffusion_risk"].to_numpy()
    z = ranked["infection_frequency"].to_numpy()
    colors = ranked["seed_infected"].to_numpy()

    fig = plt.figure(figsize=(10.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        x,
        y,
        z,
        c=colors,
        cmap="coolwarm",
        s=44,
        alpha=0.92,
        depthshade=True,
        edgecolors="#101014",
        linewidths=0.4,
    )
    for idx, row in ranked.head(8).iterrows():
        ax.text(
            idx,
            float(row["diffusion_risk"]) + 0.01,
            float(row["infection_frequency"]) + 0.012,
            str(int(row["node"])),
            fontsize=8,
            color="#121212",
        )

    ax.set_title("3D Risk Landscape from Graph Signal Results", pad=18, fontsize=15)
    ax.set_xlabel("Node rank by diffusion risk", labelpad=12)
    ax.set_ylabel("Diffusion risk score", labelpad=12)
    ax.set_zlabel("Infection frequency", labelpad=10)
    ax.xaxis.pane.set_facecolor((0.95, 0.97, 1.0, 0.88))
    ax.yaxis.pane.set_facecolor((0.98, 0.95, 0.93, 0.84))
    ax.zaxis.pane.set_facecolor((0.95, 0.98, 0.95, 0.84))
    ax.grid(True, alpha=0.2)
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.72, pad=0.08)
    colorbar.set_label("Initial infected seed", rotation=270, labelpad=18)

    def update(frame_index: int) -> list[object]:
        azim = -61 + (360.0 * frame_index / frames)
        ax.view_init(elev=26, azim=azim)
        return [scatter]

    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    _save_animation(animation, output, fps=fps)


def animate_contact_network_3d_growth(
    contacts: pd.DataFrame,
    metadata: pd.DataFrame,
    final_graph: nx.Graph,
    output: Path,
    frames: int = 24,
    fps: int = 8,
    seed: int = 7,
) -> None:
    nodes = ordered_nodes(final_graph)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    pos_2d = nx.spring_layout(final_graph, weight="weight", seed=seed, iterations=80)
    x_coords = np.array([pos_2d[node][0] for node in nodes], dtype=float)
    y_coords = np.array([pos_2d[node][1] for node in nodes], dtype=float)

    class_labels = metadata.set_index("node").reindex(nodes)["class_label"].fillna("unknown")
    unique_classes = sorted(class_labels.astype(str).unique())
    class_to_idx = {label: idx for idx, label in enumerate(unique_classes)}
    node_colors = np.array([class_to_idx[str(label)] for label in class_labels], dtype=float)

    final_weighted_degree = dict(final_graph.degree(weight="weight"))
    final_max_degree = max(final_weighted_degree.values(), default=1.0) or 1.0

    cumulative_contacts = contacts.sort_values("time").copy()
    time_points = np.quantile(
        cumulative_contacts["time"].to_numpy(dtype=float),
        np.linspace(0.04, 1.0, frames),
    )

    fig = plt.figure(figsize=(10, 7.6))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_index: int) -> list[object]:
        ax.cla()
        cutoff = float(time_points[frame_index])
        snapshot = cumulative_contacts.loc[cumulative_contacts["time"] <= cutoff]
        edge_table = (
            snapshot.groupby(["u", "v"], as_index=False)
            .agg(contact_count=("time", "size"), duration_seconds=("contact_duration_seconds", "sum"))
        )
        if edge_table.empty:
            edge_table = pd.DataFrame(columns=["u", "v", "contact_count", "duration_seconds"])

        snapshot_graph = nx.Graph()
        snapshot_graph.add_nodes_from(nodes)
        for row in edge_table.itertuples(index=False):
            duration = float(row.duration_seconds)
            snapshot_graph.add_edge(
                row.u,
                row.v,
                weight=duration,
                contact_count=int(row.contact_count),
            )

        weighted_degree = dict(snapshot_graph.degree(weight="weight"))
        z_coords = np.array(
            [weighted_degree.get(node, 0.0) / final_max_degree for node in nodes],
            dtype=float,
        )

        edge_rows = edge_table.sort_values("duration_seconds", ascending=False).head(180)
        if not edge_rows.empty:
            max_duration = max(float(edge_rows["duration_seconds"].max()), 1.0)
            for row in edge_rows.itertuples(index=False):
                idx_u = node_to_idx[row.u]
                idx_v = node_to_idx[row.v]
                ax.plot(
                    [x_coords[idx_u], x_coords[idx_v]],
                    [y_coords[idx_u], y_coords[idx_v]],
                    [0.0, 0.0],
                    color=(0.18, 0.2, 0.24, 0.12),
                    linewidth=0.2 + 1.0 * float(row.duration_seconds) / max_duration,
                )

        ax.scatter(
            x_coords,
            y_coords,
            z_coords,
            c=node_colors,
            cmap="tab20",
            s=26 + 110 * z_coords,
            alpha=0.95,
            edgecolors="#171717",
            linewidths=0.25,
            depthshade=True,
        )

        progress = (frame_index + 1) / frames
        ax.set_title(
            f"3D Growth of the Aggregated School Contact Network ({progress:.0%} of timeline)",
            pad=18,
            fontsize=14,
        )
        ax.set_xlabel("Spring-layout X", labelpad=10)
        ax.set_ylabel("Spring-layout Y", labelpad=12)
        ax.set_zlabel("Normalized cumulative contact intensity", labelpad=10)
        ax.set_xlim(x_coords.min() * 1.12, x_coords.max() * 1.12)
        ax.set_ylim(y_coords.min() * 1.12, y_coords.max() * 1.12)
        ax.set_zlim(0.0, 1.05)
        ax.view_init(elev=28, azim=-50 + frame_index * 1.25)
        ax.xaxis.pane.set_facecolor((0.95, 0.97, 1.0, 0.82))
        ax.yaxis.pane.set_facecolor((0.98, 0.95, 0.93, 0.82))
        ax.zaxis.pane.set_facecolor((0.95, 0.98, 0.95, 0.82))
        ax.grid(True, alpha=0.18)
        return []

    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    _save_animation(animation, output, fps=fps)


def plot_model_fit_curves(
    truth_counts: pd.DataFrame,
    sir_counts: pd.DataFrame,
    seir_counts: pd.DataFrame,
    output: Path,
) -> None:
    x_column = "time_step" if "time_step" in truth_counts.columns else "time"
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot(
        truth_counts[x_column],
        truth_counts["infected"],
        label="Pseudo-ground-truth SEIR",
        color="#222222",
        lw=2.6,
    )
    ax.plot(
        sir_counts[x_column] if x_column in sir_counts.columns else sir_counts["time"],
        sir_counts["infected"],
        label="Best-fit SIR",
        color="#c43c39",
        lw=2.3,
    )
    ax.plot(
        seir_counts[x_column] if x_column in seir_counts.columns else seir_counts["time"],
        seir_counts["infected"],
        label="Best-fit SEIR",
        color="#2b7a4b",
        lw=2.3,
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean infected nodes")
    ax.set_title("Model fit on infected curve")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, output)


def plot_model_metric_bars(metrics: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    models = metrics["model"].tolist()

    axes[0].bar(models, metrics["mse_infected"], color=["#c43c39", "#2b7a4b"])
    axes[0].set_title("Infected-curve MSE")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(models, metrics["mae_infected"], color=["#c43c39", "#2b7a4b"])
    axes[1].set_title("Infected-curve MAE")
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(models, metrics["curve_accuracy"], color=["#c43c39", "#2b7a4b"])
    axes[2].set_title("Curve Accuracy (1 - MAPE)")
    axes[2].set_ylim(0, 1.0)
    axes[2].grid(axis="y", alpha=0.25)

    for ax in axes:
        ax.set_xlabel("Model")
    _save(fig, output)
