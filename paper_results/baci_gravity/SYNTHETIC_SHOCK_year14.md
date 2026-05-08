# RG-A: synthetic shock injection on baci_gravity

perturbation year: snap_idx 14 (model has seen this in training)
prediction target: snap_idx 15
top 30 countries tested (by trade volume)
seeds: [0, 1, 2, 3, 4]

for each candidate country, we zero out all incident edges in the perturbation
snapshot, then predict the next snapshot. we measure prediction-error increase
for the OTHER N-1 countries (not the knocked-out one). if the model has internalized
graph structure, dropping high-volume countries should produce larger error.

## Spearman ρ between (Δprediction-error excluding self) and (knocked-out country's centrality)

| measure | median ρ across seeds | min | max |
|---|---:|---:|---:|
| volume centrality | -0.091 | -0.762 | +0.794 |
| degree centrality | -0.044 | -0.567 | +0.310 |

max delta-deviation across knockouts: median +0.0000

**reading**: ρ > 0 means dropping more-central countries causes more disruption to
the model's prediction of OTHER countries' next-snapshot embeddings. positive ρ is
evidence the model has internalized which countries are graph-structurally important.