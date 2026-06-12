from __future__ import annotations

import argparse
import copy
import math
import sys
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.model_evaluation import evaluate_curve_fit
from src.sir_simulation import monte_carlo_sir, select_initial_infected
from src.visualization import plot_model_fit_curves, plot_model_metric_bars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SEIR-Campus repo on its sample dataset and compare against weighted SIR."
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path("external") / "SEIR-Campus",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("seir_sir_comparison") / "repo_evaluation",
    )
    parser.add_argument("--truth-runs", type=int, default=80)
    parser.add_argument("--fit-runs", type=int, default=40)
    parser.add_argument("--final-runs", type=int, default=80)
    parser.add_argument("--initial-infected", type=int, default=5)
    parser.add_argument("--max-students", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def add_repo_to_path(repo_dir: Path) -> None:
    repo_path = str(repo_dir.resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def semester_to_graph(semester) -> nx.Graph:
    graph = nx.Graph()
    for student_id, demographics in semester.students.items():
        graph.add_node(student_id, demographics=demographics)

    duration_by_edge: dict[tuple[object, object], float] = {}
    for date in sorted(semester.meeting_dates):
        daily_meetings = semester.meeting_dates[date]
        for meeting, duration in daily_meetings.items():
            members = sorted(semester.meeting_enrollment[meeting], key=str)
            for u, v in combinations(members, 2):
                edge = (u, v)
                duration_by_edge[edge] = duration_by_edge.get(edge, 0.0) + float(duration)

    if not duration_by_edge:
        return graph

    max_duration = max(duration_by_edge.values())
    for (u, v), duration in duration_by_edge.items():
        weight = math.log1p(duration) / math.log1p(max_duration) if max_duration > 0 else 1.0
        graph.add_edge(u, v, duration=float(duration), weight=float(weight))
    return graph


def summary_to_dataframe(summary: dict[str, object]) -> pd.DataFrame:
    time_points = list(summary["T"])
    rows: list[dict[str, float]] = []
    for idx, date in enumerate(time_points):
        rows.append(
            {
                "time_step": float(idx),
                "susceptible": float(np.mean(summary["Ss"][date])),
                "exposed": float(np.mean(summary["Es"][date])),
                "infected": float(np.mean(summary["Is"][date])),
                "recovered": float(np.mean(summary["Rs"][date])),
                "quarantined": float(np.mean(summary["Qs"][date])),
                "vaccinated": float(np.mean(summary["Vs"][date])),
            }
        )
    return pd.DataFrame(rows)


def sample_semester(semester, max_students: int):
    if len(semester.students) <= max_students:
        return semester

    sampled = copy.deepcopy(semester)
    ranked_students = sorted(
        sampled.students,
        key=lambda student: (-len(sampled.student_enrollment[student]), str(student)),
    )
    keep = set(ranked_students[:max_students])
    remove_students = [student for student in list(sampled.students) if student not in keep]
    for student in remove_students:
        sampled.remove_student(student)

    remove_meetings = [
        meeting
        for meeting, members in list(sampled.meeting_enrollment.items())
        if len(members) < 2
    ]
    for meeting in remove_meetings:
        sampled.remove_meeting(meeting)
    sampled.clean_student_list()
    return sampled


def run_repo_seir(semester, *, repetitions: int, rate: float, initial_exposure: list[object], seed: int):
    import PySeirCampus as sc

    np.random.seed(seed)
    parameters = sc.Parameters(reps=repetitions)
    parameters.rate = rate
    parameters.daily_spontaneous_prob = 0.0
    parameters.initial_exposure = list(initial_exposure)
    parameters.start_date = min(semester.meeting_dates)
    parameters.end_date = max(semester.meeting_dates)
    _, _, _, summary = sc.run_repetitions(
        semester,
        parameters,
        report=True,
        graphics=False,
    )
    return summary_to_dataframe(summary)


def build_report_note(metrics: pd.DataFrame, repo_rate: float, sampled_students: int) -> str:
    seir = metrics.loc[metrics["model"] == "SEIR-Campus"].iloc[0]
    sir = metrics.loc[metrics["model"] == "Weighted SIR"].iloc[0]
    better_model = "SEIR-Campus" if seir["mse_infected"] < sir["mse_infected"] else "Weighted SIR"

    return "\n".join(
        [
            "# SEIR-Campus Repo Evaluation Note",
            "",
            "## Evaluation Setup",
            "",
            (
                "This evaluation uses the public sample dataset from the original `SEIR-Campus` repository. "
                "The `SEIR-Campus` simulation is executed directly from the cloned GitHub repo, and the same "
                "dataset is converted into an aggregated weighted graph for the `Weighted SIR` baseline."
            ),
            "",
            (
                f"To keep the experiment computationally feasible, the run uses a sampled subnetwork of "
                f"`{sampled_students}` students selected by enrollment activity from the original dataset."
            ),
            "",
            (
                f"The pseudo-ground-truth epidemic curve is generated by `SEIR-Campus` with "
                f"`rate={repo_rate:.8f}` and no spontaneous community exposure."
            ),
            "",
            "## Metrics",
            "",
            "- `MSE` and `MAE` compare the infected curve against the pseudo-ground-truth SEIR-Campus curve.",
            "- `Curve Accuracy = 1 - MAPE` is used as an accuracy-style fit metric.",
            "- `R^2` reports how much infected-curve variation is explained by each method.",
            "",
            "## Result Summary",
            "",
            (
                f"`SEIR-Campus` achieved `MSE={seir['mse_infected']:.3f}`, `MAE={seir['mae_infected']:.3f}`, "
                f"`Curve Accuracy={seir['curve_accuracy']:.3f}`, and `R^2={seir['r2_infected']:.3f}`."
            ),
            (
                f"`Weighted SIR` achieved `MSE={sir['mse_infected']:.3f}`, `MAE={sir['mae_infected']:.3f}`, "
                f"`Curve Accuracy={sir['curve_accuracy']:.3f}`, and `R^2={sir['r2_infected']:.3f}`."
            ),
            "",
            (
                f"Under this dataset-driven evaluation, `{better_model}` fits the SEIR-Campus outbreak trajectory better. "
                "This is consistent with the fact that SEIR-Campus explicitly models the exposed stage and day-by-day contact structure."
            ),
            "",
            "## Reporting Note",
            "",
            (
                "These values should be presented as outbreak-curve fitting metrics, not classifier accuracy, "
                "because both methods are epidemic simulators."
            ),
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    add_repo_to_path(args.repo_dir)
    import PySeirCampus as sc

    args.output_dir.mkdir(parents=True, exist_ok=True)

    holiday_list = [(2020, 10, 14)]
    holidays = set(sc.datetime(*h) for h in holiday_list)
    semester = sc.Semester(str(args.repo_dir / "publicdata.data"), holidays)
    semester = sample_semester(semester, args.max_students)
    graph = semester_to_graph(semester)
    initial_infected = select_initial_infected(
        graph,
        count=args.initial_infected,
        seed=args.seed,
        strategy="high_degree",
    )

    base_rate = 1 / (65.6 * 60 * 7)
    truth_counts = run_repo_seir(
        semester,
        repetitions=args.truth_runs,
        rate=base_rate,
        initial_exposure=initial_infected,
        seed=args.seed,
    )

    seir_grid = [base_rate * scale for scale in [0.7, 0.85, 1.0, 1.15, 1.3]]
    sir_betas = [0.006, 0.008, 0.010, 0.012, 0.015]
    sir_mus = [0.10, 0.14, 0.18, 0.22]

    search_rows: list[dict[str, float | str]] = []

    best_repo_rate = None
    best_repo_counts = None
    best_repo_metrics = None
    best_repo_loss = None
    for index, rate in enumerate(seir_grid):
        counts = run_repo_seir(
            semester,
            repetitions=args.fit_runs,
            rate=rate,
            initial_exposure=initial_infected,
            seed=args.seed + 100 + index,
        )
        metrics = evaluate_curve_fit(
            truth_counts,
            counts,
            parameter_count=1,
        )
        search_rows.append({"model": "SEIR-Campus", "rate": rate, **metrics})
        if best_repo_loss is None or metrics["mse_infected"] < best_repo_loss:
            best_repo_loss = metrics["mse_infected"]
            best_repo_rate = rate
            best_repo_counts = counts
            best_repo_metrics = metrics

    best_sir_params = None
    best_sir_counts = None
    best_sir_metrics = None
    best_sir_loss = None
    for beta in sir_betas:
        for mu in sir_mus:
            counts = monte_carlo_sir(
                graph=graph,
                beta=beta,
                mu=mu,
                steps=len(truth_counts) - 1,
                runs=args.fit_runs,
                initial_infected=initial_infected,
                seed=args.seed + int(beta * 10000) + int(mu * 1000),
            )["mean_counts"].rename(columns={"time": "time_step"})
            metrics = evaluate_curve_fit(
                truth_counts,
                counts,
                parameter_count=2,
            )
            search_rows.append({"model": "Weighted SIR", "beta": beta, "mu": mu, **metrics})
            if best_sir_loss is None or metrics["mse_infected"] < best_sir_loss:
                best_sir_loss = metrics["mse_infected"]
                best_sir_params = {"beta": beta, "mu": mu}
                best_sir_counts = counts
                best_sir_metrics = metrics

    final_repo_counts = run_repo_seir(
        semester,
        repetitions=args.final_runs,
        rate=float(best_repo_rate),
        initial_exposure=initial_infected,
        seed=args.seed + 5000,
    )
    final_repo_metrics = evaluate_curve_fit(
        truth_counts,
        final_repo_counts,
        parameter_count=1,
    )

    final_sir_counts = monte_carlo_sir(
        graph=graph,
        beta=float(best_sir_params["beta"]),
        mu=float(best_sir_params["mu"]),
        steps=len(truth_counts) - 1,
        runs=args.final_runs,
        initial_infected=initial_infected,
        seed=args.seed + 6000,
    )["mean_counts"].rename(columns={"time": "time_step"})
    final_sir_metrics = evaluate_curve_fit(
        truth_counts,
        final_sir_counts,
        parameter_count=2,
    )

    metrics_df = pd.DataFrame(
        [
            {"model": "SEIR-Campus", "rate": float(best_repo_rate), **final_repo_metrics},
            {"model": "Weighted SIR", **best_sir_params, **final_sir_metrics},
        ]
    )
    pd.DataFrame(search_rows).to_csv(args.output_dir / "grid_search_results.csv", index=False)
    truth_counts.to_csv(args.output_dir / "truth_seir_campus_curve.csv", index=False)
    final_repo_counts.to_csv(args.output_dir / "seir_campus_best_fit_curve.csv", index=False)
    final_sir_counts.to_csv(args.output_dir / "weighted_sir_best_fit_curve.csv", index=False)
    metrics_df.to_csv(args.output_dir / "model_comparison_metrics.csv", index=False)

    plot_model_fit_curves(
        truth_counts,
        final_sir_counts,
        final_repo_counts,
        args.output_dir / "model_fit_curves.png",
    )
    plot_model_metric_bars(
        pd.DataFrame(
            [
                {"model": "Weighted SIR", **final_sir_metrics},
                {"model": "SEIR-Campus", **final_repo_metrics},
            ]
        ),
        args.output_dir / "model_metric_bars.png",
    )

    note = build_report_note(metrics_df, base_rate, graph.number_of_nodes())
    (args.output_dir / "report_note.md").write_text(note, encoding="utf-8")
    print(f"Outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
