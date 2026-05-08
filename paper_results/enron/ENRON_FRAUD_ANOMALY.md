# E-P: Enron per-person fraud-week anomaly grouped by role

hypothesis: people doing role-relevant crisis response (legal, govaffairs)
show higher prediction-error anomaly during fraud weeks than admin staff.

fraud snaps: [100, 101, 104, 107]
baseline: non-fraud test snaps in [101, 119]
seeds: [0, 1, 2, 3, 4]

## graph

| role | n | mean fraud-week anomaly | median |
|---|---:|---:|---:|
| trading | 5 | +0.1927 | +0.1948 |
| exec | 3 | +0.0662 | -0.1260 |
| research | 1 | +0.0204 | +0.0204 |
| admin | 13 | -0.1170 | +0.0001 |
| legal | 8 | -0.1507 | -0.0728 |
| operations | 2 | -0.1885 | -0.1885 |
| govaffairs | 6 | -0.3115 | -0.3452 |

MWU one-sided p (legal+govaffairs > admin): 0.8341

## sequential

| role | n | mean fraud-week anomaly | median |
|---|---:|---:|---:|
| trading | 5 | +0.1223 | +0.0669 |
| research | 1 | -0.0020 | -0.0020 |
| exec | 3 | -0.0260 | -0.0866 |
| admin | 13 | -0.0858 | -0.0079 |
| legal | 8 | -0.1094 | -0.0505 |
| operations | 2 | -0.1265 | -0.1265 |
| govaffairs | 6 | -0.2584 | -0.2647 |

MWU one-sided p (legal+govaffairs > admin): 0.8217
