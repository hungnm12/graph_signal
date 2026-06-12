# SEIR vs SIR Loss and Accuracy Note

## Evaluation Setup

The comparison uses the same weighted contact graph and the same initial infected nodes for both models. Because the repository does not include labeled outbreak observations, a Monte Carlo `SIR` simulation is used as pseudo-ground-truth.

Pseudo-ground-truth parameters: `beta=0.012`, `mu=0.140`.

## Metrics

- `MSE` and `MAE` measure the difference between the predicted infected curve and the pseudo-ground-truth infected curve.
- `Curve Accuracy = 1 - MAPE` is used as an accuracy-style metric for time-series fitting.
- `R^2` measures how well each model explains the infected-curve variance.
- `AIC` and `BIC` penalize extra free parameters, which matters on an aggregated graph where the exposed state is not directly observed.

## Result Summary

`SEIR` achieved `MSE=37.307 ± 7.309`, `MAE=5.422 ± 0.603`, `Curve Accuracy=0.665 ± 0.027`, and `R^2=0.916 ± 0.016`.
`SIR` achieved `MSE=19.315 ± 11.143`, `MAE=3.397 ± 1.141`, `Curve Accuracy=0.863 ± 0.070`, `R^2=0.957 ± 0.025`, `AIC=148.723 ± 26.798`, and `BIC=152.587 ± 26.798`.

Using `bic_infected` as the model-selection criterion, `SIR` is preferred under this evaluation.
For the aggregated contact graph, this criterion is more defensible than raw fit alone because the data collapse away the temporal detail needed to identify an exposed compartment cleanly.

## Files

- `model_comparison_metrics.csv`: final loss and accuracy values.
- `grid_search_results.csv`: parameter-search results for both models.
- `model_fit_curves.png`: infected-curve fit figure.
- `model_metric_bars.png`: bar chart of loss and accuracy metrics.

## Reporting Note

In the report, these metrics should be described as model-fitting metrics rather than classification accuracy, because `SEIR` and `SIR` are epidemic simulators instead of classifiers.
