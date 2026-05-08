# anomaly trajectory: icio

per-snapshot prediction deviation (1 - mean_pred_cos), median across seeds [0, 1, 2, 3, 4].
shock years are flagged with ★. graph-vs-sequential gap on shock years tells us whether graph-JEPA picks up real-world dynamics more cleanly than the non-graph baseline.

| year | snap idx | graph dev (median) | seq dev (median) | shock? | label |
|---|---:|---:|---:|:-:|---|
| 2004 | 4 | 0.0131 | 0.0450 |  |  |
| 2005 | 5 | 0.0263 | 0.0996 |  |  |
| 2006 | 6 | 0.0085 | 0.0342 |  |  |
| 2007 | 7 | 0.0217 | 0.0577 |  |  |
| 2008 | 8 | 0.0170 | 0.0607 | ★ | Global Financial Crisis |
| 2009 | 9 | 0.0090 | 0.0203 | ★ | GFC trough |
| 2010 | 10 | 0.0108 | 0.0237 |  |  |
| 2011 | 11 | 0.0160 | 0.0359 |  |  |
| 2012 | 12 | 0.0295 | 0.0693 |  |  |
| 2013 | 13 | 0.0232 | 0.0834 |  |  |
| 2014 | 14 | 0.0172 | 0.0492 |  |  |

## shock detection score

- **graph**: non-shock median dev = 0.0172, shock median dev = 0.0130, lift = -0.0042 (0.76× baseline)
- **sequential**: non-shock median dev = 0.0492, shock median dev = 0.0405, lift = -0.0087 (0.82× baseline)