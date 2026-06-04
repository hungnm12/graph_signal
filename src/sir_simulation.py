from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd

from .load_graph import ordered_nodes


SUSCEPTIBLE = 0
INFECTED = 1
RECOVERED = 2
VACCINATED = 3


@dataclass(frozen=True)
class SimulationRun:
    counts: pd.DataFrame
    infected_signal: np.ndarray
    ever_infected_signal: np.ndarray
    metrics: dict[str, float]


def select_initial_infected(
    graph: nx.Graph,
    count: int,
    seed: int = 42,
    exclude: Iterable[object] | None = None,
    strategy: str = "high_degree",
) -> list[object]:
    excluded = set(exclude or [])
    candidates = [node for node in ordered_nodes(graph) if node not in excluded]
    if len(candidates) < count:
        raise ValueError("Not enough non-excluded nodes to select initial infections.")

    if strategy == "random":
        rng = np.random.default_rng(seed)
        return list(rng.choice(candidates, size=count, replace=False))

    weighted_degree = dict(graph.degree(weight="weight"))
    candidates.sort(key=lambda node: (-weighted_degree.get(node, 0.0), str(node)))
    return candidates[:count]


def run_sir(
    graph: nx.Graph,
    beta: float,
    mu: float,
    steps: int,
    initial_infected: Iterable[object],
    vaccinated: Iterable[object] | None = None,
    seed: int | None = None,
) -> SimulationRun:
    rng = np.random.default_rng(seed)
    nodes = ordered_nodes(graph)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)

    states = np.full(n, SUSCEPTIBLE, dtype=np.int8)
    vaccinated_set = set(vaccinated or [])
    for node in vaccinated_set:
        if node in node_to_idx:
            states[node_to_idx[node]] = VACCINATED

    initial = [node for node in initial_infected if node in node_to_idx and node not in vaccinated_set]
    if not initial:
        raise ValueError("Initial infected set is empty after excluding vaccinated nodes.")

    for node in initial:
        states[node_to_idx[node]] = INFECTED

    counts_rows: list[dict[str, float]] = []
    infected_history = np.zeros((steps + 1, n), dtype=float)
    ever_infected = states == INFECTED

    def record(t: int) -> None:
        infected_history[t] = states == INFECTED
        counts_rows.append(
            {
                "time": float(t),
                "susceptible": float(np.count_nonzero(states == SUSCEPTIBLE)),
                "infected": float(np.count_nonzero(states == INFECTED)),
                "recovered": float(np.count_nonzero(states == RECOVERED)),
                "vaccinated": float(np.count_nonzero(states == VACCINATED)),
            }
        )

    record(0)
    for t in range(1, steps + 1):
        infected_indices = np.flatnonzero(states == INFECTED)
        new_infections: set[int] = set()
        recoveries: set[int] = set()

        for u_idx in infected_indices:
            u = nodes[u_idx]
            for v in graph.neighbors(u):
                v_idx = node_to_idx[v]
                if states[v_idx] != SUSCEPTIBLE:
                    continue
                weight = float(graph[u][v].get("weight", 1.0))
                p_transmit = 1.0 - np.exp(-beta * weight)
                if rng.random() < p_transmit:
                    new_infections.add(v_idx)

            if rng.random() < mu:
                recoveries.add(u_idx)

        if recoveries:
            states[list(recoveries)] = RECOVERED
        if new_infections:
            states[list(new_infections)] = INFECTED
            ever_infected[list(new_infections)] = True

        record(t)

    counts = pd.DataFrame(counts_rows)
    peak_idx = int(counts["infected"].idxmax())
    unvaccinated = max(1, n - len(vaccinated_set))
    metrics = {
        "final_infected_ratio": float(ever_infected.sum() / n),
        "attack_rate_unvaccinated": float(ever_infected.sum() / unvaccinated),
        "peak_infected_count": float(counts.loc[peak_idx, "infected"]),
        "time_to_peak": float(counts.loc[peak_idx, "time"]),
        "total_vaccinated": float(len(vaccinated_set)),
    }

    return SimulationRun(
        counts=counts,
        infected_signal=infected_history,
        ever_infected_signal=ever_infected.astype(float),
        metrics=metrics,
    )


def monte_carlo_sir(
    graph: nx.Graph,
    beta: float,
    mu: float,
    steps: int,
    runs: int,
    initial_infected: Iterable[object],
    vaccinated: Iterable[object] | None = None,
    seed: int = 42,
) -> dict[str, object]:
    nodes = ordered_nodes(graph)
    n = len(nodes)
    counts_sum: pd.DataFrame | None = None
    infected_signal_sum = np.zeros((steps + 1, n), dtype=float)
    ever_infected_sum = np.zeros(n, dtype=float)
    metric_rows: list[dict[str, float]] = []

    for run_idx in range(runs):
        result = run_sir(
            graph=graph,
            beta=beta,
            mu=mu,
            steps=steps,
            initial_infected=initial_infected,
            vaccinated=vaccinated,
            seed=seed + run_idx,
        )
        if counts_sum is None:
            counts_sum = result.counts.copy()
        else:
            for column in ["susceptible", "infected", "recovered", "vaccinated"]:
                counts_sum[column] += result.counts[column]
        infected_signal_sum += result.infected_signal
        ever_infected_sum += result.ever_infected_signal
        metric_rows.append(result.metrics)

    if counts_sum is None:
        raise ValueError("runs must be at least 1")

    mean_counts = counts_sum.copy()
    for column in ["susceptible", "infected", "recovered", "vaccinated"]:
        mean_counts[column] /= runs

    metrics = pd.DataFrame(metric_rows)
    summary = metrics.mean(numeric_only=True).to_dict()
    for column in metrics.columns:
        summary[f"{column}_std"] = float(metrics[column].std(ddof=1))

    return {
        "nodes": nodes,
        "mean_counts": mean_counts,
        "mean_infected_signal": infected_signal_sum / runs,
        "infection_frequency": ever_infected_sum / runs,
        "metrics": metrics,
        "summary": summary,
    }

