# SEIR-Campus vs Weighted SIR

## Summary

The graph shows that the `SEIR` model delays the outbreak peak because individuals first move into the exposed (`E`) state before becoming infectious. By contrast, the `SIR` model moves directly from susceptible to infectious, so the infection curve typically rises earlier and more sharply.

For a campus setting, `SEIR-Campus` is more realistic because it captures incubation and time-dependent disease progression. `Weighted SIR` is still useful as a simpler baseline because it is easier to simulate, easier to interpret, and easier to connect with graph-based metrics such as degree, centrality, and spectral radius.

In short, `SEIR-Campus` is stronger for realistic outbreak modeling and intervention analysis, while `Weighted SIR` is stronger for transparent comparison and graph signal processing interpretation.

## Asset

- Raw SIR graph: [sir_raw_curves.png](./sir_raw_curves.png)
- Raw SEIR graph: [seir_raw_curves.png](./seir_raw_curves.png)
- Comparison graph: [seir_vs_sir_curves.png](./seir_vs_sir_curves.png)
