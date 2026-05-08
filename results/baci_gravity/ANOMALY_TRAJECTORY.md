# anomaly trajectory: baci_gravity

per-snapshot prediction deviation (1 - mean_pred_cos), median across seeds [0, 1, 2, 3, 4].
shock years are flagged with ★. graph-vs-sequential gap on shock years tells us whether graph-JEPA picks up real-world dynamics more cleanly than the non-graph baseline.

| year | snap idx | graph dev (median) | seq dev (median) | shock? | label |
|---|---:|---:|---:|:-:|---|
| 2000 | 4 | 0.0997 | 0.3041 |  |  |
| 2001 | 5 | 0.0968 | 0.2928 | ★ | Dot-com / 9-11 |
| 2002 | 6 | 0.0830 | 0.2786 |  |  |
| 2003 | 7 | 0.0663 | 0.2623 |  |  |
| 2004 | 8 | 0.0716 | 0.2353 |  |  |
| 2005 | 9 | 0.0863 | 0.2464 |  |  |
| 2006 | 10 | 0.0556 | 0.2525 |  |  |
| 2007 | 11 | 0.0662 | 0.2503 |  |  |
| 2008 | 12 | 0.0635 | 0.2507 | ★ | Global Financial Crisis |
| 2009 | 13 | 0.0609 | 0.2525 | ★ | GFC trough |
| 2010 | 14 | 0.0447 | 0.2476 |  |  |
| 2011 | 15 | 0.0666 | 0.2856 |  |  |
| 2012 | 16 | 0.0479 | 0.2602 |  |  |
| 2013 | 17 | 0.0456 | 0.2715 |  |  |
| 2014 | 18 | 0.0623 | 0.2450 | ★ | Oil price collapse |
| 2015 | 19 | 0.0705 | 0.2592 | ★ | Oil collapse / Russia sanctions |
| 2016 | 20 | 0.0510 | 0.2611 |  |  |
| 2017 | 21 | 0.0471 | 0.2472 |  |  |
| 2018 | 22 | 0.0428 | 0.2488 | ★ | US-China trade war |
| 2019 | 23 | 0.0449 | 0.2510 |  |  |
| 2020 | 24 | 0.0767 | 0.2706 | ★ | COVID-19 |

## shock detection score

- **graph**: non-shock median dev = 0.0609, shock median dev = 0.0635, lift = +0.0027 (1.04× baseline)
- **sequential**: non-shock median dev = 0.2563, shock median dev = 0.2525, lift = -0.0039 (0.98× baseline)