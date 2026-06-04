from __future__ import annotations

import networkx as nx
import numpy as np


def _top_nodes_by_scores(scores: dict[object, float], count: int) -> set[object]:
    ranked = sorted(scores, key=lambda node: (-scores[node], str(node)))
    return set(ranked[:count])


def random_vaccination(graph: nx.Graph, fraction: float, seed: int = 42) -> set[object]:
    rng = np.random.default_rng(seed)
    nodes = list(graph.nodes())
    count = max(1, int(round(fraction * len(nodes))))
    return set(rng.choice(nodes, size=count, replace=False))


def degree_vaccination(graph: nx.Graph, fraction: float) -> set[object]:
    count = max(1, int(round(fraction * graph.number_of_nodes())))
    scores = dict(graph.degree(weight="weight"))
    return _top_nodes_by_scores(scores, count)


def eigenvector_vaccination(graph: nx.Graph, fraction: float) -> set[object]:
    count = max(1, int(round(fraction * graph.number_of_nodes())))
    try:
        scores = nx.eigenvector_centrality_numpy(graph, weight="weight")
    except Exception:
        scores = nx.eigenvector_centrality(graph, weight="weight", max_iter=1000)
    return _top_nodes_by_scores(scores, count)


def scale_edge_weights(graph: nx.Graph, scale: float) -> nx.Graph:
    modified = graph.copy()
    for u, v, data in modified.edges(data=True):
        data["weight"] = float(data.get("weight", 1.0)) * scale
    return modified


def remove_random_contacts(graph: nx.Graph, fraction: float, seed: int = 42) -> nx.Graph:
    modified = graph.copy()
    edges = list(modified.edges())
    remove_count = int(round(fraction * len(edges)))
    if remove_count <= 0:
        return modified

    rng = np.random.default_rng(seed)
    selected = rng.choice(len(edges), size=remove_count, replace=False)
    modified.remove_edges_from([edges[idx] for idx in selected])
    return modified


def build_intervention_scenarios(
    graph: nx.Graph,
    vaccination_fraction: float = 0.10,
    seed: int = 42,
) -> list[dict[str, object]]:
    return [
        {
            "name": "Baseline",
            "graph": graph,
            "vaccinated": set(),
        },
        {
            "name": "Random vaccination",
            "graph": graph,
            "vaccinated": random_vaccination(graph, vaccination_fraction, seed=seed),
        },
        {
            "name": "Degree vaccination",
            "graph": graph,
            "vaccinated": degree_vaccination(graph, vaccination_fraction),
        },
        {
            "name": "Eigenvector vaccination",
            "graph": graph,
            "vaccinated": eigenvector_vaccination(graph, vaccination_fraction),
        },
        {
            "name": "Mask edge reduction",
            "graph": scale_edge_weights(graph, 0.50),
            "vaccinated": set(),
        },
        {
            "name": "Contact reduction",
            "graph": remove_random_contacts(graph, 0.30, seed=seed),
            "vaccinated": set(),
        },
    ]

