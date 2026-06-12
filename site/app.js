const summary = window.RESULT_SUMMARY;
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
