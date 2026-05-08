# Enron role-recovery — advanced analyses

Source: paper_results/enron/seed0/role_recovery.json
Date: 2026-05-08

## E2: sub-cluster discovery within each role (k-means k=2)

If a role has internal substructure the encoder picked up on, k-means k=2
should split it cleanly. The interesting case is govaffairs: it anti-clustered
in the aggregate (gap −0.62). If sub-clusters split California-regulatory vs
federal/general govaffairs people, the negative gap was actually the encoder
discovering finer-grained structure than the hand-curated labels.

### graph encoder

**govaffairs** (n=6)

- Sub-cluster A (n=2, within-cos 1.000): ['christi.nicolay@enron.com', 'james.steffes@enron.com']
- Sub-cluster B (n=4, within-cos 0.915): ['jeff.dasovich@enron.com', 'susan.mara@enron.com', 'richard.shapiro@enron.com', 'alan.comnes@enron.com']

**admin** (n=13)

- Sub-cluster A (n=9, within-cos 0.968): ['rhonda.denton@enron.com', 'lorna.brennan@enron.com', 'janette.elbertson@enron.com', 'julie.clyatt@enron.com', 'miyung.buster@enron.com', 'janet.butler@enron.com', 'rosalee.fleming@enron.com', 'taffy.milligan@enron.com', 'tamara.black@enron.com']
- Sub-cluster B (n=4, within-cos 0.812): ['veronica.espinoza@enron.com', 'cheryl.johnson@enron.com', 'ginger.dernehl@enron.com', 'susan.bailey@enron.com']

**trading** (n=5)

- Sub-cluster A (n=2, within-cos 0.316): ['chris.germany@enron.com', 'eric.bass@enron.com']
- Sub-cluster B (n=3, within-cos 0.821): ['david.forster@enron.com', 'kate.symes@enron.com', 'mike.grigsby@enron.com']

**legal** (n=8)

- Sub-cluster A (n=2, within-cos 1.000): ['mary.hain@enron.com', 'mark.taylor@enron.com']
- Sub-cluster B (n=6, within-cos 0.948): ['kay.mann@enron.com', 'sara.shackleton@enron.com', 'stephanie.panus@enron.com', 'mary.cook@enron.com', 'gerald.nemec@enron.com', 'sarah.novosel@enron.com']

### sequential encoder

**govaffairs** (n=6)

- Sub-cluster A (n=2, within-cos 0.999): ['christi.nicolay@enron.com', 'james.steffes@enron.com']
- Sub-cluster B (n=4, within-cos 0.726): ['jeff.dasovich@enron.com', 'susan.mara@enron.com', 'richard.shapiro@enron.com', 'alan.comnes@enron.com']

**admin** (n=13)

- Sub-cluster A (n=9, within-cos 0.964): ['rhonda.denton@enron.com', 'lorna.brennan@enron.com', 'janette.elbertson@enron.com', 'julie.clyatt@enron.com', 'miyung.buster@enron.com', 'janet.butler@enron.com', 'rosalee.fleming@enron.com', 'taffy.milligan@enron.com', 'tamara.black@enron.com']
- Sub-cluster B (n=4, within-cos 0.665): ['veronica.espinoza@enron.com', 'cheryl.johnson@enron.com', 'ginger.dernehl@enron.com', 'susan.bailey@enron.com']

**trading** (n=5)

- Sub-cluster A (n=2, within-cos 0.678): ['chris.germany@enron.com', 'eric.bass@enron.com']
- Sub-cluster B (n=3, within-cos 0.929): ['david.forster@enron.com', 'kate.symes@enron.com', 'mike.grigsby@enron.com']

**legal** (n=8)

- Sub-cluster A (n=2, within-cos 1.000): ['mary.hain@enron.com', 'mark.taylor@enron.com']
- Sub-cluster B (n=6, within-cos 0.953): ['kay.mann@enron.com', 'sara.shackleton@enron.com', 'stephanie.panus@enron.com', 'mary.cook@enron.com', 'gerald.nemec@enron.com', 'sarah.novosel@enron.com']

## E3: role-shuffle null control

For each condition, role labels are randomly permuted 1000 times and the within-vs-between cosine gap recomputed. The null distribution is the gap that would arise from random labeling. If the actual observed gap is in the right tail (shuffle_p ≈ 0), the real role labels carry signal beyond chance. If shuffle_p ≈ 0.5, the signal could plausibly arise from random labels — i.e. an artifact.

| condition | actual gap | null mean | null p95 | shuffle p (one-sided) |
|---|---:|---:|---:|---:|
| graph | +0.815 | -0.073 | +0.856 | 0.0580 |
| sequential | +0.686 | -0.008 | +0.499 | 0.0180 |
| raw_features | -0.001 | -0.004 | +0.058 | 0.4710 |

**Reading**: `shuffle_p < 0.05` means the actual gap is in the top 5% of the random-label null — i.e. the role labels are doing real work. `shuffle_p ≈ 0.5` means the gap could be reproduced by random label assignment — the signal is artifact.