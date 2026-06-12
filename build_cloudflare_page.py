from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from src.load_graph import _sorted_pair, prepare_contact_graph, read_temporal_contacts
from src.visualization import (
    animate_contact_network_3d_growth,
    animate_risk_scatter_3d_rotation,
    plot_risk_scatter_3d,
)


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
SITE_DIR = ROOT / "site"
ASSETS_DIR = SITE_DIR / "assets"


def copy_static_assets() -> None:
    assets = {
        "graph_visualization.png": RESULTS_DIR / "graph_visualization.png",
        "intervention_comparison.png": RESULTS_DIR / "intervention_comparison.png",
        "infection_rate_sweep.png": RESULTS_DIR / "infection_rate_sweep.png",
        "risk_prediction_scatter.png": RESULTS_DIR / "risk_prediction_scatter.png",
        "smoothness_curve.png": RESULTS_DIR / "smoothness_curve.png",
    }
    for name, source in assets.items():
        shutil.copy2(source, ASSETS_DIR / name)


def load_contacts_for_animation() -> tuple[pd.DataFrame, pd.DataFrame, object]:
    graph, _, metadata, _ = prepare_contact_graph(data_dir=ROOT / "data")
    contacts = read_temporal_contacts(ROOT / "data" / "raw" / "primaryschool.csv.gz").copy()
    pairs = [_sorted_pair(row.node_i, row.node_j) for row in contacts.itertuples(index=False)]
    contacts["u"] = [pair[0] for pair in pairs]
    contacts["v"] = [pair[1] for pair in pairs]
    contacts["contact_duration_seconds"] = 20
    return contacts, metadata, graph


def build_summary_payload() -> dict[str, object]:
    project_summary = pd.read_csv(RESULTS_DIR / "project_summary.csv").iloc[0]
    intervention_summary = pd.read_csv(RESULTS_DIR / "intervention_summary.csv")
    beta_summary = pd.read_csv(RESULTS_DIR / "beta_sweep_summary.csv")

    best_intervention = intervention_summary.sort_values("final_infected_ratio").iloc[0]
    highest_beta = beta_summary.sort_values("final_infected_ratio", ascending=False).iloc[0]

    return {
        "headline": {
            "source": str(project_summary["data_source"]),
            "nodes": int(project_summary["nodes"]),
            "edges": int(project_summary["edges"]),
            "spectral_radius": round(float(project_summary["largest_adjacency_eigenvalue"]), 2),
        },
        "baseline": {
            "final_infected_ratio": round(float(project_summary["baseline_final_infected_ratio"]) * 100, 2),
            "peak_infected_count": round(float(project_summary["baseline_peak_infected_count"]), 2),
            "time_to_peak": round(float(project_summary["baseline_time_to_peak"]), 2),
        },
        "risk_quality": {
            "pearson": round(float(project_summary["risk_pearson"]), 3),
            "spearman": round(float(project_summary["risk_spearman"]), 3),
            "precision_at_k": round(float(project_summary["risk_precision_at_k"]), 3),
        },
        "best_intervention": {
            "scenario": str(best_intervention["scenario"]),
            "final_infected_ratio": round(float(best_intervention["final_infected_ratio"]) * 100, 2),
            "peak_infected_count": round(float(best_intervention["peak_infected_count"]), 2),
        },
        "beta_extreme": {
            "beta": round(float(highest_beta["beta"]), 4),
            "final_infected_ratio": round(float(highest_beta["final_infected_ratio"]) * 100, 2),
            "time_to_peak": round(float(highest_beta["time_to_peak"]), 2),
        },
    }


def write_site_files(summary: dict[str, object]) -> None:
    html = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Graph Signal Result Viewer</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet" />
    <link rel="stylesheet" href="./styles.css" />
  </head>
  <body>
    <main class="page">
      <section class="hero">
        <div class="hero-copy">
          <p class="eyebrow">Cloudflare Worker Page</p>
          <h1>3D epidemic risk graph built directly from simulation results.</h1>
          <p class="lede">
            This page packages the school contact-network outputs into a publishable result viewer.
            The hero visualization is a real 3D scatter generated from <code>results/risk_scores.csv</code>.
          </p>
          <div class="metric-grid" id="metric-grid"></div>
        </div>
        <div class="hero-visual">
          <img src="./assets/risk_scores_3d.png" alt="3D risk landscape graph" />
        </div>
      </section>

      <section class="panel">
        <div class="panel-heading">
          <p class="eyebrow">Animated 3D Views</p>
          <h2>Motion studies for the school contact network</h2>
        </div>
        <div class="gallery">
          <figure>
            <img src="./assets/contact_network_growth_3d.gif" alt="3D animated school contact network growth" />
            <figcaption>Aggregated graph growth over the temporal contact stream</figcaption>
          </figure>
          <figure>
            <img src="./assets/risk_scores_3d_rotate.gif" alt="360 degree rotating 3D graph signal results" />
            <figcaption>360-degree rotation of the Graph Signal 3D risk landscape</figcaption>
          </figure>
        </div>
      </section>

      <section class="panel">
        <div class="panel-heading">
          <p class="eyebrow">Result Gallery</p>
          <h2>Supporting views from the pipeline</h2>
        </div>
        <div class="gallery">
          <figure>
            <img src="./assets/graph_visualization.png" alt="Graph visualization" />
            <figcaption>Aggregated school contact graph</figcaption>
          </figure>
          <figure>
            <img src="./assets/intervention_comparison.png" alt="Intervention comparison" />
            <figcaption>Intervention comparison</figcaption>
          </figure>
          <figure>
            <img src="./assets/infection_rate_sweep.png" alt="Infection rate sweep" />
            <figcaption>Beta sweep impact</figcaption>
          </figure>
          <figure>
            <img src="./assets/risk_prediction_scatter.png" alt="Risk prediction scatter" />
            <figcaption>Risk score vs infection frequency</figcaption>
          </figure>
          <figure>
            <img src="./assets/smoothness_curve.png" alt="Smoothness curve" />
            <figcaption>Graph-signal smoothness over time</figcaption>
          </figure>
        </div>
      </section>
    </main>
    <script>
      window.RESULT_SUMMARY = __SUMMARY_JSON__;
    </script>
    <script src="./app.js"></script>
  </body>
</html>
"""
    css = """:root {
  --bg: #f4efe7;
  --panel: rgba(255, 250, 244, 0.8);
  --ink: #171717;
  --muted: #5b534b;
  --line: rgba(23, 23, 23, 0.12);
  --accent: #c65d3a;
  --accent-soft: #f5c9a7;
  --shadow: 0 24px 80px rgba(44, 31, 16, 0.12);
}

* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Space Grotesk", sans-serif;
  color: var(--ink);
  background:
    radial-gradient(circle at top left, rgba(245, 201, 167, 0.55), transparent 32%),
    radial-gradient(circle at right 20%, rgba(102, 141, 255, 0.18), transparent 28%),
    linear-gradient(180deg, #f8f1e8 0%, #efe6da 100%);
}

.page {
  width: min(1200px, calc(100vw - 32px));
  margin: 0 auto;
  padding: 32px 0 48px;
}

.hero,
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 28px;
  box-shadow: var(--shadow);
  backdrop-filter: blur(16px);
}

.hero {
  display: grid;
  grid-template-columns: 1.1fr 0.95fr;
  gap: 28px;
  align-items: center;
  padding: 28px;
}

.eyebrow {
  margin: 0 0 10px;
  color: var(--accent);
  letter-spacing: 0.14em;
  text-transform: uppercase;
  font-size: 12px;
  font-weight: 700;
}

h1,
h2 {
  margin: 0;
  line-height: 0.96;
  font-weight: 700;
}

h1 {
  font-size: clamp(2.8rem, 5vw, 5.7rem);
  max-width: 12ch;
}

h2 {
  font-size: clamp(2rem, 4vw, 3.2rem);
}

.lede {
  max-width: 62ch;
  color: var(--muted);
  font-size: 1rem;
  line-height: 1.7;
  margin: 18px 0 0;
}

.hero-visual img,
.gallery img {
  width: 100%;
  display: block;
  border-radius: 20px;
  border: 1px solid rgba(255, 255, 255, 0.65);
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
  margin-top: 22px;
}

.metric-card {
  background: rgba(255, 255, 255, 0.68);
  border: 1px solid rgba(23, 23, 23, 0.08);
  border-radius: 18px;
  padding: 16px;
}

.metric-card strong {
  display: block;
  font-size: 1.8rem;
  margin-top: 4px;
}

.metric-card span {
  color: var(--muted);
  font-size: 0.92rem;
}

.panel {
  margin-top: 22px;
  padding: 24px;
}

.panel-heading {
  margin-bottom: 18px;
}

.gallery {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
}

figure {
  margin: 0;
}

figcaption {
  margin-top: 10px;
  color: var(--muted);
  font-size: 0.92rem;
}

code {
  font-family: inherit;
  background: rgba(23, 23, 23, 0.06);
  padding: 0.15rem 0.38rem;
  border-radius: 999px;
}

@media (max-width: 900px) {
  .hero,
  .gallery {
    grid-template-columns: 1fr;
  }

  .metric-grid {
    grid-template-columns: 1fr;
  }

  .page {
    width: min(100vw - 20px, 1200px);
    padding-top: 10px;
  }

  .hero,
  .panel {
    border-radius: 22px;
  }
}
"""
    js = """const summary = window.RESULT_SUMMARY;
const metrics = [
  {
    label: "Network size",
    value: `${summary.headline.nodes} nodes / ${summary.headline.edges} edges`,
    hint: summary.headline.source,
  },
  {
    label: "Baseline spread",
    value: `${summary.baseline.final_infected_ratio}%`,
    hint: `Peak ${summary.baseline.peak_infected_count} at t=${summary.baseline.time_to_peak}`,
  },
  {
    label: "Best intervention",
    value: summary.best_intervention.scenario,
    hint: `${summary.best_intervention.final_infected_ratio}% final infected ratio`,
  },
  {
    label: "Risk correlation",
    value: `${summary.risk_quality.pearson} Pearson`,
    hint: `Spearman ${summary.risk_quality.spearman}, P@20 ${summary.risk_quality.precision_at_k}`,
  },
];

document.getElementById("metric-grid").innerHTML = metrics
  .map(
    (metric) => `
      <article class="metric-card">
        <span>${metric.label}</span>
        <strong>${metric.value}</strong>
        <span>${metric.hint}</span>
      </article>
    `
  )
  .join("");
"""

    SITE_DIR.mkdir(parents=True, exist_ok=True)
    (SITE_DIR / "index.html").write_text(
        html.replace("__SUMMARY_JSON__", json.dumps(summary, ensure_ascii=True)),
        encoding="utf-8",
    )
    (SITE_DIR / "styles.css").write_text(css, encoding="utf-8")
    (SITE_DIR / "app.js").write_text(js, encoding="utf-8")


def main() -> None:
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    risk_scores = pd.read_csv(RESULTS_DIR / "risk_scores.csv")
    plot_risk_scatter_3d(risk_scores, ASSETS_DIR / "risk_scores_3d.png")
    animate_risk_scatter_3d_rotation(risk_scores, ASSETS_DIR / "risk_scores_3d_rotate.gif")
    contacts, metadata, graph = load_contacts_for_animation()
    animate_contact_network_3d_growth(
        contacts,
        metadata,
        graph,
        ASSETS_DIR / "contact_network_growth_3d.gif",
    )
    copy_static_assets()
    summary = build_summary_payload()
    write_site_files(summary)

    print(f"Cloudflare site generated at: {SITE_DIR}")


if __name__ == "__main__":
    main()
