from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import networkx as nx
import numpy as np
import pandas as pd

from .seir_simulation import monte_carlo_seir
from .sir_simulation import monte_carlo_sir


ModelRunner = Callable[..., dict[str, object]]


@dataclass(frozen=True)
class FitResult:
    model_name: str
    params: dict[str, float]
    mean_counts: pd.DataFrame
    metrics: dict[str, float]


def _information_criteria(
    truth_values: np.ndarray,
    pred_values: np.ndarray,
    parameter_count: int,
) -> dict[str, float]:
    n = int(len(truth_values))
    if n == 0:
        return {
            "rss_infected": 0.0,
            "aic_infected": float("nan"),
            "bic_infected": float("nan"),
        }

    residual = pred_values - truth_values
    rss = float(np.sum(residual ** 2))
    mse_floor = max(rss / n, 1e-12)
    aic = float(n * np.log(mse_floor) + 2 * parameter_count)
    bic = float(n * np.log(mse_floor) + parameter_count * np.log(n))
    return {
        "rss_infected": rss,
        "aic_infected": aic,
        "bic_infected": bic,
    }


def _curve_metrics(
    truth: pd.Series,
    prediction: pd.Series,
    parameter_count: int,
) -> dict[str, float]:
    truth_values = truth.to_numpy(dtype=float)
    pred_values = prediction.to_numpy(dtype=float)
    error = pred_values - truth_values
    mse = float(np.mean(error ** 2))
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(mse))

    denominator = np.where(np.abs(truth_values) < 1e-8, 1.0, truth_values)
    mape = float(np.mean(np.abs(error) / denominator))
    curve_accuracy = float(max(0.0, 1.0 - mape))

    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((truth_values - truth_values.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    peak_pred_idx = int(np.argmax(pred_values))
    peak_true_idx = int(np.argmax(truth_values))
    peak_error = float(abs(pred_values[peak_pred_idx] - truth_values[peak_true_idx]))
    time_to_peak_error = float(abs(peak_pred_idx - peak_true_idx))
    final_size_error = float(abs(pred_values[-1] - truth_values[-1]))

    return {
        "mse_infected": mse,
        "mae_infected": mae,
        "rmse_infected": rmse,
        "mape_infected": mape,
        "curve_accuracy": curve_accuracy,
        "r2_infected": float(r2),
        "peak_infected_error": peak_error,
        "time_to_peak_error": time_to_peak_error,
        "final_infected_error": final_size_error,
        **_information_criteria(truth_values, pred_values, parameter_count),
    }


def evaluate_curve_fit(
    truth_counts: pd.DataFrame,
    predicted_counts: pd.DataFrame,
    *,
    parameter_count: int,
) -> dict[str, float]:
    metrics = _curve_metrics(
        truth_counts["infected"],
        predicted_counts["infected"],
        parameter_count,
    )
    if "recovered" in truth_counts and "recovered" in predicted_counts:
        recovered_metrics = _curve_metrics(
            truth_counts["recovered"],
            predicted_counts["recovered"],
            parameter_count,
        )
        metrics["mse_recovered"] = recovered_metrics["mse_infected"]
        metrics["mae_recovered"] = recovered_metrics["mae_infected"]
        metrics["r2_recovered"] = recovered_metrics["r2_infected"]
    return metrics


def _fit_model(
    *,
    model_name: str,
    graph: nx.Graph,
    truth_counts: pd.DataFrame,
    initial_infected: list[object],
    steps: int,
    runs: int,
    seed: int,
    runner: ModelRunner,
    grid: dict[str, list[float]],
    selection_metric: str,
) -> tuple[FitResult, pd.DataFrame]:
    rows: list[dict[str, float | str]] = []
    best_result: FitResult | None = None
    best_loss: float | None = None

    keys = list(grid.keys())
    for index, values in enumerate(product(*(grid[key] for key in keys))):
        params = {key: float(value) for key, value in zip(keys, values)}
        run = runner(
            graph=graph,
            steps=steps,
            runs=runs,
            initial_infected=initial_infected,
            seed=seed + index * 17,
            **params,
        )
        mean_counts = run["mean_counts"]
        fit_metrics = evaluate_curve_fit(
            truth_counts,
            mean_counts,
            parameter_count=len(params),
        )
        row: dict[str, float | str] = {"model": model_name, **params, **fit_metrics}
        rows.append(row)

        score = float(fit_metrics[selection_metric])
        if best_loss is None or score < best_loss:
            best_loss = score
            best_result = FitResult(
                model_name=model_name,
                params=params,
                mean_counts=mean_counts,
                metrics=fit_metrics,
            )

    if best_result is None:
        raise ValueError(f"No fit results produced for {model_name}.")

    return best_result, pd.DataFrame(rows)


def run_model_comparison(
    graph: nx.Graph,
    initial_infected: list[object],
    *,
    steps: int = 60,
    truth_runs: int = 140,
    fit_runs: int = 80,
    final_runs: int = 140,
    evaluation_repeats: int = 6,
    seed: int = 42,
    truth_model: str = "sir",
    selection_metric: str = "bic_infected",
) -> dict[str, object]:
    if truth_model == "sir":
        truth_params = {"beta": 0.012, "mu": 0.14}
        truth_run = monte_carlo_sir(
            graph=graph,
            beta=truth_params["beta"],
            mu=truth_params["mu"],
            steps=steps,
            runs=truth_runs,
            initial_infected=initial_infected,
            seed=seed,
        )
    elif truth_model == "seir":
        truth_params = {"beta": 0.022, "sigma": 0.28, "gamma": 0.18}
        truth_run = monte_carlo_seir(
            graph=graph,
            beta=truth_params["beta"],
            sigma=truth_params["sigma"],
            gamma=truth_params["gamma"],
            steps=steps,
            runs=truth_runs,
            initial_infected=initial_infected,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported truth_model: {truth_model}")
    truth_counts = truth_run["mean_counts"].copy()

    sir_grid = {
        "beta": [0.012, 0.016, 0.020, 0.024, 0.028],
        "mu": [0.10, 0.14, 0.18, 0.22],
    }
    seir_grid = {
        "beta": [0.016, 0.018, 0.020, 0.022, 0.024, 0.028],
        "sigma": [0.28, 0.34, 0.50, 0.75, 1.00],
        "gamma": [0.14, 0.18, 0.22],
    }

    best_sir_fit, sir_search = _fit_model(
        model_name="SIR",
        graph=graph,
        truth_counts=truth_counts,
        initial_infected=initial_infected,
        steps=steps,
        runs=fit_runs,
        seed=seed + 1000,
        runner=monte_carlo_sir,
        grid=sir_grid,
        selection_metric=selection_metric,
    )
    best_seir_fit, seir_search = _fit_model(
        model_name="SEIR",
        graph=graph,
        truth_counts=truth_counts,
        initial_infected=initial_infected,
        steps=steps,
        runs=fit_runs,
        seed=seed + 2000,
        runner=monte_carlo_seir,
        grid=seir_grid,
        selection_metric=selection_metric,
    )

    def summarize_trials(model_name: str, runner: ModelRunner, params: dict[str, float], base_seed: int) -> tuple[dict[str, float], pd.DataFrame]:
        metric_rows: list[dict[str, float]] = []
        trial_counts: list[pd.DataFrame] = []
        for repeat_idx in range(evaluation_repeats):
            result = runner(
                graph=graph,
                steps=steps,
                runs=final_runs,
                initial_infected=initial_infected,
                seed=base_seed + repeat_idx * 97,
                **params,
            )
            counts = result["mean_counts"]
            trial_counts.append(counts)
            metric_rows.append(
                evaluate_curve_fit(
                    truth_counts,
                    counts,
                    parameter_count=len(params),
                )
            )

        metrics_frame = pd.DataFrame(metric_rows)
        summary = metrics_frame.mean(numeric_only=True).to_dict()
        for column in metrics_frame.columns:
            summary[f"{column}_std"] = float(metrics_frame[column].std(ddof=1))
        summary["evaluation_repeats"] = float(evaluation_repeats)
        summary["model"] = model_name
        return summary, trial_counts[0]

    sir_metrics, sir_counts = summarize_trials(
        "SIR",
        monte_carlo_sir,
        best_sir_fit.params,
        seed + 3000,
    )
    seir_metrics, seir_counts = summarize_trials(
        "SEIR",
        monte_carlo_seir,
        best_seir_fit.params,
        seed + 4000,
    )
    metrics_df = pd.DataFrame(
        [
            {
                "truth_model": truth_model.upper(),
                "selection_metric": selection_metric,
                **best_sir_fit.params,
                **sir_metrics,
            },
            {
                "truth_model": truth_model.upper(),
                "selection_metric": selection_metric,
                **best_seir_fit.params,
                **seir_metrics,
            },
        ]
    )

    return {
        "truth_params": truth_params,
        "truth_model": truth_model.upper(),
        "selection_metric": selection_metric,
        "truth_counts": truth_counts,
        "sir_counts": sir_counts,
        "seir_counts": seir_counts,
        "metrics": metrics_df,
        "grid_search": pd.concat([sir_search, seir_search], ignore_index=True),
    }
