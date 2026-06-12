# Predicting and Controlling Epidemic Spreading on School Contact Networks


## What is included

- Aggregates a school face-to-face contact dataset into a weighted contact graph.
- Runs Monte Carlo SIR simulation on the weighted graph.
- Adds a SEIR-vs-SIR comparison workflow for report figures and model-fitting metrics.
- Sweeps infection rate values.
- Compares intervention strategies:
  - random vaccination
  - degree-based vaccination
  - eigenvector-centrality vaccination
  - mask-style edge-weight reduction
  - random contact reduction
- Computes Graph Signal Processing outputs:
  - weighted adjacency and graph Laplacian
  - largest adjacency eigenvalue
  - infection graph-signal smoothness `x(t)^T L x(t)`
  - heat-diffusion risk score `exp(-tau L) x0` with default `tau=0.05`
  - correlation and precision@20 between diffusion risk and Monte Carlo infection frequency

## Data

The default pipeline downloads the SocioPatterns primary school temporal contact
dataset:

https://www.sociopatterns.org/datasets/primary-school-temporal-network-data/

Direct file used by the code:

https://sociopatterns.org/assets/data/primaryschool.csv.gz

The data are distributed by SocioPatterns under a Creative Commons
Attribution-NonCommercial-ShareAlike license. Cite the SocioPatterns dataset and
the papers listed in the final project guide when using the results in slides,
poster, or report.

If the download fails, the code creates a synthetic school contact network so
the project remains runnable offline. The summary file records whether the
synthetic fallback was used.

For the SEIR comparison workflow, the repository also supports running against
the public sample data from the original `SEIR-Campus` GitHub project after the
repo is cloned locally.

## How to run

From this folder:

```powershell
py -3 -m pip install -r requirements.txt
py -3 main.py
```

To avoid downloading data and force the synthetic fallback:

```powershell
py -3 main.py --no-download
```

Useful parameters:

```powershell
py -3 main.py --beta 0.005 --mu 0.16 --steps 80 --runs 120
```

To generate the SEIR-vs-SIR comparison graphs in `seir_sir_comparison/`:

```powershell
py -3 seir_sir_comparison/generate_comparison_plot.py
```

To run the dataset-driven `SEIR-Campus` vs `Weighted SIR` evaluation:

```powershell
py -3 run_repo_model_comparison.py --max-students 250 --truth-runs 6 --fit-runs 3 --final-runs 6
```

This script expects the original `SEIR-Campus` repository at
`external/SEIR-Campus`.

## Outputs

CSV files in `results/`:

- `project_summary.csv`
- `graph_statistics.csv`
- `baseline_summary.csv`
- `baseline_sir_curve.csv`
- `beta_sweep_summary.csv`
- `intervention_summary.csv`
- `risk_scores.csv`
- `risk_prediction_metrics.csv`
- `smoothness_over_time.csv`

Figures in `results/`:

- `graph_visualization.png`
- `sir_curve.png`
- `infection_rate_sweep.png`
- `intervention_comparison.png`
- `diffusion_risk_score.png`
- `risk_prediction_scatter.png`
- `smoothness_curve.png`

Additional comparison outputs in `seir_sir_comparison/repo_evaluation/`:

- `model_comparison_metrics.csv`
- `grid_search_results.csv`
- `model_fit_curves.png`
- `model_metric_bars.png`
- `report_note.md`

## Cloudflare Worker page

This repository now includes a static result page that is generated from the
files in `results/` and served by a minimal Cloudflare Worker.

Build the page locally:

```bash
python3 build_cloudflare_page.py
```

This writes the deployable site to `site/`, including:

- `site/assets/risk_scores_3d.png`: a 3D scatter built from `results/risk_scores.csv`
- existing pipeline figures copied into `site/assets/`
- `site/index.html`, `site/styles.css`, and `site/app.js`

To deploy with Wrangler:

```bash
export CLOUDFLARE_API_TOKEN=...
npm install
npm run deploy:cloudflare
```

The Worker configuration lives in `wrangler.toml`, and static assets are served
through `cloudflare/worker.js`.

## Suggested slide or report structure

1. Motivation: classroom and campus transmission can be studied as network spreading.
2. Graph formulation: nodes are students/teachers, weighted edges are cumulative contact duration.
3. SIR model: infection probability is `1 - exp(-beta * w_ij)`, recovery probability is `mu`.
4. GSP analysis: graph Laplacian, smoothness, heat diffusion risk, spectral radius.
5. Results: baseline SIR curve, beta sweep, intervention comparison, diffusion risk plot.
6. Discussion: graph structure concentrates risk around highly connected nodes; targeted vaccination should reduce outbreak size more than random vaccination when contact heterogeneity is high.
