from __future__ import annotations

import gzip
import math
import urllib.request
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd


PRIMARY_SCHOOL_URL = (
    "https://sociopatterns.org/assets/data/primaryschool.csv.gz"
)
CONTACT_COLUMNS = ["time", "node_i", "node_j", "class_i", "class_j"]


def _node_sort_key(node: object) -> tuple[int, object]:
    try:
        return (0, int(node))
    except (TypeError, ValueError):
        return (1, str(node))


def ordered_nodes(graph: nx.Graph) -> list[object]:
    return sorted(graph.nodes(), key=_node_sort_key)


def download_primary_school_dataset(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    output = raw_dir / "primaryschool.csv.gz"
    if not output.exists():
        urllib.request.urlretrieve(PRIMARY_SCHOOL_URL, output)
    return output


def _read_first_line(path: Path) -> str:
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt"
    with opener(path, mode, encoding="utf-8", errors="replace") as handle:
        return handle.readline().strip()


def read_temporal_contacts(path: Path) -> pd.DataFrame:
    first_line = _read_first_line(path).lower()
    has_header = any(token in first_line for token in ("time", "node_i", "class_i"))

    if has_header:
        df = pd.read_csv(path, sep=r"[\t, ]+", engine="python", compression="infer")
        rename_map = {
            "t": "time",
            "i": "node_i",
            "j": "node_j",
            "ci": "class_i",
            "cj": "class_j",
            "Ci": "class_i",
            "Cj": "class_j",
        }
        df = df.rename(columns=rename_map)
    else:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            names=CONTACT_COLUMNS,
            engine="python",
            compression="infer",
        )

    missing = set(CONTACT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected contact columns: {sorted(missing)}")

    df = df[CONTACT_COLUMNS].dropna()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    return df


def generate_synthetic_contacts(output: Path, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    classes = ["1A", "1B", "2A", "2B", "3A", "3B"]
    class_size = 24
    nodes: list[tuple[str, str]] = []
    for class_label in classes:
        for idx in range(class_size):
            nodes.append((f"{class_label}_{idx:02d}", class_label))

    rows: list[dict[str, object]] = []
    slots = 180
    for slot in range(slots):
        t = 20 * (slot + 1)
        for class_label in classes:
            class_nodes = [node for node, cls in nodes if cls == class_label]
            for _ in range(26):
                u, v = rng.choice(class_nodes, size=2, replace=False)
                rows.append(
                    {
                        "time": t,
                        "node_i": u,
                        "node_j": v,
                        "class_i": class_label,
                        "class_j": class_label,
                    }
                )

        for _ in range(14):
            (u, cu), (v, cv) = rng.choice(nodes, size=2, replace=False)
            if cu == cv:
                continue
            rows.append(
                {
                    "time": t,
                    "node_i": u,
                    "node_j": v,
                    "class_i": cu,
                    "class_j": cv,
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def _sorted_pair(u: object, v: object) -> tuple[object, object]:
    if _node_sort_key(u) <= _node_sort_key(v):
        return u, v
    return v, u


def aggregate_contacts(contacts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = [_sorted_pair(row.node_i, row.node_j) for row in contacts.itertuples(index=False)]
    contacts = contacts.copy()
    contacts["u"] = [pair[0] for pair in pairs]
    contacts["v"] = [pair[1] for pair in pairs]
    contacts["contact_duration_seconds"] = 20

    edges = (
        contacts.groupby(["u", "v"], as_index=False)
        .agg(
            contact_count=("time", "size"),
            duration_seconds=("contact_duration_seconds", "sum"),
            first_time=("time", "min"),
            last_time=("time", "max"),
        )
        .sort_values(["u", "v"])
    )

    max_duration = float(edges["duration_seconds"].max())
    if max_duration <= 0:
        edges["weight"] = 1.0
    else:
        edges["weight"] = np.log1p(edges["duration_seconds"]) / math.log1p(max_duration)

    left_meta = contacts[["node_i", "class_i"]].rename(
        columns={"node_i": "node", "class_i": "class_label"}
    )
    right_meta = contacts[["node_j", "class_j"]].rename(
        columns={"node_j": "node", "class_j": "class_label"}
    )
    metadata = (
        pd.concat([left_meta, right_meta], ignore_index=True)
        .drop_duplicates(subset=["node"])
        .sort_values("node", key=lambda series: series.astype(str))
        .reset_index(drop=True)
    )

    return edges.reset_index(drop=True), metadata


def graph_from_edges(edges: pd.DataFrame, metadata: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for row in metadata.itertuples(index=False):
        graph.add_node(row.node, class_label=str(row.class_label))

    for row in edges.itertuples(index=False):
        graph.add_edge(
            row.u,
            row.v,
            weight=float(row.weight),
            duration_seconds=float(row.duration_seconds),
            contact_count=int(row.contact_count),
        )

    return graph


def prepare_contact_graph(
    data_dir: Path,
    allow_download: bool = True,
    force_download: bool = False,
    seed: int = 42,
) -> tuple[nx.Graph, pd.DataFrame, pd.DataFrame, str]:
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / "primaryschool.csv.gz"
    synthetic_path = raw_dir / "synthetic_primaryschool_contacts.csv"

    source = "SocioPatterns primary school temporal network"
    if force_download and raw_path.exists():
        raw_path.unlink()

    if raw_path.exists():
        contact_path = raw_path
    elif allow_download:
        try:
            contact_path = download_primary_school_dataset(raw_dir)
        except Exception as exc:
            contact_path = generate_synthetic_contacts(synthetic_path, seed=seed)
            source = f"Synthetic fallback graph because download failed: {exc}"
    else:
        contact_path = generate_synthetic_contacts(synthetic_path, seed=seed)
        source = "Synthetic fallback graph; download disabled"

    contacts = read_temporal_contacts(contact_path)
    edges, metadata = aggregate_contacts(contacts)
    graph = graph_from_edges(edges, metadata)

    edges.to_csv(data_dir / "contact_network.csv", index=False)
    metadata.to_csv(data_dir / "node_metadata.csv", index=False)
    return graph, edges, metadata, source


def graph_statistics(graph: nx.Graph) -> dict[str, float]:
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    degrees = np.array([degree for _, degree in graph.degree()], dtype=float)
    weighted_degrees = np.array([degree for _, degree in graph.degree(weight="weight")])

    components = list(nx.connected_components(graph))
    largest_component = max((len(component) for component in components), default=0)

    return {
        "nodes": float(n),
        "edges": float(m),
        "density": float(nx.density(graph)) if n > 1 else 0.0,
        "average_degree": float(degrees.mean()) if n else 0.0,
        "average_weighted_degree": float(weighted_degrees.mean()) if n else 0.0,
        "largest_component_nodes": float(largest_component),
        "connected_components": float(len(components)),
    }
