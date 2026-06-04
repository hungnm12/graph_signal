from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.graph_signal_analysis import (
    evaluate_risk_prediction,
    graph_matrices,
    heat_diffusion_risk,
    smoothness_over_time,
    spectral_summary,
)
from src.interventions import build_intervention_scenarios
from src.load_graph import graph_statistics, prepare_contact_graph
from src.sir_simulation import monte_carlo_sir, select_initial_infected
from src.visualization import (
    plot_beta_sweep,
    plot_diffusion_risk,
    plot_graph,
    plot_intervention_comparison,
    plot_risk_scatter,
    plot_sir_curve,
    plot_smoothness,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GSP final project: epidemic spreading on school contact networks."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--beta", type=float, default=0.005)
    parser.add_argument("--mu", type=float, default=0.16)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--runs", type=int, default=120)
    parser.add_argument("--initial-infected", type=int, default=3)
    parser.add_argument("--vaccination-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def _scenario_initial_nodes(graph, count: int, seed: int, vaccinated: set[object]) -> list[object]:
    return select_initial_infected(
        graph,
        count=count,
        seed=seed,
        exclude=vaccinated,
        strategy="high_degree",
    )


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    graph, edges, metadata, source = prepare_contact_graph(
        data_dir=args.data_dir,
        allow_download=not args.no_download,
        force_download=args.force_download,
        seed=args.seed,
    )

    initial_infected = select_initial_infected(
        graph,
        count=args.initial_infected,
        seed=args.seed,
        strategy="high_degree",
    )

    baseline = monte_carlo_sir(
        graph=graph,
        beta=args.beta,
        mu=args.mu,
        steps=args.steps,
        runs=args.runs,
        initial_infected=initial_infected,
        seed=args.seed,
    )

    graph_stats = graph_statistics(graph)
    graph_stats.update(spectral_summary(graph))
    graph_stats["source_is_synthetic"] = float("Synthetic" in source)
    pd.DataFrame([graph_stats]).to_csv(args.results_dir / "graph_statistics.csv", index=False)

    nodes, adjacency, laplacian = graph_matrices(graph)
    smoothness = smoothness_over_time(baseline["mean_infected_signal"], laplacian)
    smoothness.to_csv(args.results_dir / "smoothness_over_time.csv", index=False)

    risk_scores = heat_diffusion_risk(graph, initial_infected=initial_infected, tau=0.05)
    risk_scores["infection_frequency"] = baseline["infection_frequency"]
    risk_scores.to_csv(args.results_dir / "risk_scores.csv", index=False)
    risk_eval = evaluate_risk_prediction(
        risk_scores,
        infection_frequency=baseline["infection_frequency"],
        initial_infected=initial_infected,
        k=20,
    )
    pd.DataFrame([risk_eval]).to_csv(args.results_dir / "risk_prediction_metrics.csv", index=False)

    baseline["mean_counts"].to_csv(args.results_dir / "baseline_sir_curve.csv", index=False)
    pd.DataFrame([baseline["summary"]]).to_csv(args.results_dir / "baseline_summary.csv", index=False)

    beta_rows: list[dict[str, float]] = []
    for beta in [args.beta * 0.5, args.beta, args.beta * 1.5]:
        run = monte_carlo_sir(
            graph=graph,
            beta=beta,
            mu=args.mu,
            steps=args.steps,
            runs=max(40, args.runs // 2),
            initial_infected=initial_infected,
            seed=args.seed + int(beta * 10000),
        )
        row = {"beta": beta, "mu": args.mu, "nodes": graph.number_of_nodes()}
        row.update(run["summary"])
        beta_rows.append(row)
    beta_summary = pd.DataFrame(beta_rows)
    beta_summary.to_csv(args.results_dir / "beta_sweep_summary.csv", index=False)

    intervention_rows: list[dict[str, float | str]] = []
    for scenario in build_intervention_scenarios(
        graph,
        vaccination_fraction=args.vaccination_fraction,
        seed=args.seed,
    ):
        scenario_graph = scenario["graph"]
        vaccinated = set(scenario["vaccinated"])
        scenario_initial = _scenario_initial_nodes(
            scenario_graph,
            args.initial_infected,
            args.seed,
            vaccinated,
        )
        run = monte_carlo_sir(
            graph=scenario_graph,
            beta=args.beta,
            mu=args.mu,
            steps=args.steps,
            runs=max(40, args.runs // 2),
            initial_infected=scenario_initial,
            vaccinated=vaccinated,
            seed=args.seed + len(intervention_rows) * 1000,
        )
        row = {
            "scenario": str(scenario["name"]),
            "initial_infected": ",".join(str(node) for node in scenario_initial),
        }
        row.update(run["summary"])
        intervention_rows.append(row)
    intervention_summary = pd.DataFrame(intervention_rows)
    intervention_summary.to_csv(args.results_dir / "intervention_summary.csv", index=False)

    summary = {
        "data_source": source,
        "initial_infected": ",".join(str(node) for node in initial_infected),
        "beta": args.beta,
        "mu": args.mu,
        "runs": args.runs,
        "steps": args.steps,
        **graph_stats,
        **{f"risk_{key}": value for key, value in risk_eval.items()},
        **{f"baseline_{key}": value for key, value in baseline["summary"].items()},
    }
    pd.DataFrame([summary]).to_csv(args.results_dir / "project_summary.csv", index=False)

    plot_graph(graph, args.results_dir / "graph_visualization.png", seed=args.seed)
    plot_sir_curve(
        baseline["mean_counts"],
        args.results_dir / "sir_curve.png",
        title="Baseline weighted-graph SIR simulation",
    )
    plot_beta_sweep(beta_summary, args.results_dir / "infection_rate_sweep.png")
    plot_intervention_comparison(
        intervention_summary,
        args.results_dir / "intervention_comparison.png",
    )
    plot_diffusion_risk(
        graph,
        risk_scores,
        initial_infected=initial_infected,
        output=args.results_dir / "diffusion_risk_score.png",
        seed=args.seed,
    )
    plot_risk_scatter(risk_scores, args.results_dir / "risk_prediction_scatter.png")
    plot_smoothness(smoothness, args.results_dir / "smoothness_curve.png")

    print("GSP final project pipeline completed.")
    print(f"Data source: {source}")
    print(f"Nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")
    print(f"Initial infected nodes: {', '.join(str(node) for node in initial_infected)}")
    print(f"Results written to: {args.results_dir.resolve()}")


if __name__ == "__main__":
    main()
