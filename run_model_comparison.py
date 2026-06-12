from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.load_graph import prepare_contact_graph
from src.model_evaluation import run_model_comparison
from src.sir_simulation import select_initial_infected
from src.visualization import plot_model_fit_curves, plot_model_metric_bars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SEIR vs SIR loss/accuracy comparison outputs."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("seir_sir_comparison") / "evaluation",
    )
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--truth-runs", type=int, default=140)
    parser.add_argument("--fit-runs", type=int, default=80)
    parser.add_argument("--final-runs", type=int, default=140)
    parser.add_argument("--evaluation-repeats", type=int, default=6)
    parser.add_argument("--initial-infected", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--truth-model",
        choices=["sir", "seir"],
        default="sir",
        help="Pseudo-ground-truth family used to generate the reference outbreak curve.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["mse_infected", "aic_infected", "bic_infected"],
        default="bic_infected",
        help="Criterion used during parameter search. Lower is better.",
    )
    return parser.parse_args()


def build_report_note(
    metrics: pd.DataFrame,
    truth_params: dict[str, float],
    truth_model: str,
    selection_metric: str,
) -> str:
    sir = metrics.loc[metrics["model"] == "SIR"].iloc[0]
    seir = metrics.loc[metrics["model"] == "SEIR"].iloc[0]

    better_model = "SEIR" if seir[selection_metric] < sir[selection_metric] else "SIR"
    truth_param_text = ", ".join(f"`{key}={value:.3f}`" for key, value in truth_params.items())
    return "\n".join(
        [
            "# SEIR vs SIR Loss and Accuracy Note",
            "",
            "## Evaluation Setup",
            "",
            (
                "The comparison uses the same weighted contact graph and the same initial infected nodes "
                "for both models. Because the repository does not include labeled outbreak observations, "
                f"a Monte Carlo `{truth_model}` simulation is used as pseudo-ground-truth."
            ),
            "",
            f"Pseudo-ground-truth parameters: {truth_param_text}.",
            "",
            "## Metrics",
            "",
            "- `MSE` and `MAE` measure the difference between the predicted infected curve and the pseudo-ground-truth infected curve.",
            "- `Curve Accuracy = 1 - MAPE` is used as an accuracy-style metric for time-series fitting.",
            "- `R^2` measures how well each model explains the infected-curve variance.",
            "- `AIC` and `BIC` penalize extra free parameters, which matters on an aggregated graph where the exposed state is not directly observed.",
            "",
            "## Result Summary",
            "",
            (
                f"`SEIR` achieved `MSE={seir['mse_infected']:.3f} ± {seir['mse_infected_std']:.3f}`, "
                f"`MAE={seir['mae_infected']:.3f} ± {seir['mae_infected_std']:.3f}`, "
                f"`Curve Accuracy={seir['curve_accuracy']:.3f} ± {seir['curve_accuracy_std']:.3f}`, "
                f"and `R^2={seir['r2_infected']:.3f} ± {seir['r2_infected_std']:.3f}`."
            ),
            (
                f"`SIR` achieved `MSE={sir['mse_infected']:.3f} ± {sir['mse_infected_std']:.3f}`, "
                f"`MAE={sir['mae_infected']:.3f} ± {sir['mae_infected_std']:.3f}`, "
                f"`Curve Accuracy={sir['curve_accuracy']:.3f} ± {sir['curve_accuracy_std']:.3f}`, "
                f"`R^2={sir['r2_infected']:.3f} ± {sir['r2_infected_std']:.3f}`, "
                f"`AIC={sir['aic_infected']:.3f} ± {sir['aic_infected_std']:.3f}`, and "
                f"`BIC={sir['bic_infected']:.3f} ± {sir['bic_infected_std']:.3f}`."
            ),
            "",
            (
                f"Using `{selection_metric}` as the model-selection criterion, `{better_model}` is preferred under this evaluation."
            ),
            (
                "For the aggregated contact graph, this criterion is more defensible than raw fit alone because "
                "the data collapse away the temporal detail needed to identify an exposed compartment cleanly."
            ),
            "",
            "## Files",
            "",
            "- `model_comparison_metrics.csv`: final loss and accuracy values.",
            "- `grid_search_results.csv`: parameter-search results for both models.",
            "- `model_fit_curves.png`: infected-curve fit figure.",
            "- `model_metric_bars.png`: bar chart of loss and accuracy metrics.",
            "",
            "## Reporting Note",
            "",
            (
                "In the report, these metrics should be described as model-fitting metrics rather than "
                "classification accuracy, because `SEIR` and `SIR` are epidemic simulators instead of classifiers."
            ),
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    graph, _, _, _ = prepare_contact_graph(
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

    comparison = run_model_comparison(
        graph,
        initial_infected,
        steps=args.steps,
        truth_runs=args.truth_runs,
        fit_runs=args.fit_runs,
        final_runs=args.final_runs,
        evaluation_repeats=args.evaluation_repeats,
        seed=args.seed,
        truth_model=args.truth_model,
        selection_metric=args.selection_metric,
    )

    truth_counts = comparison["truth_counts"]
    sir_counts = comparison["sir_counts"]
    seir_counts = comparison["seir_counts"]
    metrics = comparison["metrics"]
    grid_search = comparison["grid_search"]

    truth_counts.to_csv(args.output_dir / "truth_seir_curve.csv", index=False)
    sir_counts.to_csv(args.output_dir / "sir_best_fit_curve.csv", index=False)
    seir_counts.to_csv(args.output_dir / "seir_best_fit_curve.csv", index=False)
    metrics.to_csv(args.output_dir / "model_comparison_metrics.csv", index=False)
    grid_search.to_csv(args.output_dir / "grid_search_results.csv", index=False)

    plot_model_fit_curves(
        truth_counts,
        sir_counts,
        seir_counts,
        args.output_dir / "model_fit_curves.png",
    )
    plot_model_metric_bars(metrics, args.output_dir / "model_metric_bars.png")

    note = build_report_note(
        metrics,
        comparison["truth_params"],
        comparison["truth_model"],
        comparison["selection_metric"],
    )
    (args.output_dir / "report_note.md").write_text(note, encoding="utf-8")

    print(f"Outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
