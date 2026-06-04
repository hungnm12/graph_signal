from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.stats import rankdata, spearmanr

from .load_graph import ordered_nodes


def graph_matrices(graph: nx.Graph) -> tuple[list[object], np.ndarray, np.ndarray]:
    nodes = ordered_nodes(graph)
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight="weight", dtype=float)
    degree = np.diag(adjacency.sum(axis=1))
    laplacian = degree - adjacency
    return nodes, adjacency, laplacian


def largest_eigenvalue(adjacency: np.ndarray) -> float:
    if adjacency.size == 0:
        return 0.0
    eigenvalues = np.linalg.eigvalsh(adjacency)
    return float(eigenvalues[-1])


def graph_smoothness(signal: np.ndarray, laplacian: np.ndarray) -> float:
    x = np.asarray(signal, dtype=float).reshape(-1)
    return float(x.T @ laplacian @ x)


def smoothness_over_time(
    mean_infected_signal: np.ndarray, laplacian: np.ndarray
) -> pd.DataFrame:
    rows = []
    for t, signal in enumerate(mean_infected_signal):
        rows.append(
            {
                "time": t,
                "smoothness": graph_smoothness(signal, laplacian),
                "mean_infection_probability": float(np.mean(signal)),
            }
        )
    return pd.DataFrame(rows)


def heat_diffusion_risk(
    graph: nx.Graph,
    initial_infected: list[object],
    tau: float = 0.05,
) -> pd.DataFrame:
    nodes, _, laplacian = graph_matrices(graph)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    x0 = np.zeros(len(nodes), dtype=float)
    for node in initial_infected:
        if node in node_to_idx:
            x0[node_to_idx[node]] = 1.0

    diffused = expm_multiply(csr_matrix(-tau * laplacian), x0)
    min_value = float(diffused.min()) if len(diffused) else 0.0
    max_value = float(diffused.max()) if len(diffused) else 0.0
    if max_value > min_value:
        risk = (diffused - min_value) / (max_value - min_value)
    else:
        risk = np.zeros_like(diffused)

    return pd.DataFrame({"node": nodes, "diffusion_risk": risk, "seed_infected": x0})


def evaluate_risk_prediction(
    risk_scores: pd.DataFrame,
    infection_frequency: np.ndarray,
    initial_infected: list[object],
    k: int = 20,
) -> dict[str, float]:
    scores = risk_scores.copy()
    scores["infection_frequency"] = infection_frequency
    mask = ~scores["node"].isin(initial_infected)
    evaluation = scores.loc[mask].copy()

    if evaluation.empty:
        return {"pearson": float("nan"), "spearman": float("nan"), "precision_at_k": 0.0}

    x = evaluation["diffusion_risk"].to_numpy(dtype=float)
    y = evaluation["infection_frequency"].to_numpy(dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(x, y)[0, 1])

    spearman_value = spearmanr(x, y).statistic
    spearman = float(spearman_value) if np.isfinite(spearman_value) else float("nan")

    k = min(k, len(evaluation))
    predicted_top = set(evaluation.nlargest(k, "diffusion_risk")["node"])
    actual_top = set(evaluation.nlargest(k, "infection_frequency")["node"])
    precision_at_k = len(predicted_top & actual_top) / k if k else 0.0

    return {
        "pearson": pearson,
        "spearman": spearman,
        "precision_at_k": float(precision_at_k),
    }


def spectral_summary(graph: nx.Graph) -> dict[str, float]:
    _, adjacency, laplacian = graph_matrices(graph)
    return {
        "largest_adjacency_eigenvalue": largest_eigenvalue(adjacency),
        "laplacian_trace": float(np.trace(laplacian)),
        "total_weight": float(adjacency.sum() / 2.0),
    }
