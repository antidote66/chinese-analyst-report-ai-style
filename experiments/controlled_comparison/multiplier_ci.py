"""Recompute the §5.5 multipliers and 95% bootstrap CIs from paired_word_stats.csv (no report text needed).

Reproduces cross_model_results.csv exactly: same estimator as cross_model.py
(ratio = mean S-group occurrence of model texts / mean of paired 2019 human texts;
2,000 report-level bootstrap draws; np.random.seed(0) set once, conditions in sorted order).
"""
import numpy as np, pandas as pd, os
here = os.path.dirname(os.path.abspath(__file__))
P = pd.read_csv(os.path.join(here, 'paired_word_stats.csv'), dtype={'rptid': str})
np.random.seed(0); out = []
for name in sorted(P.condition_file.unique()):
    d = P[P.condition_file == name]; g = d.S_model.values; h = d.S_human.values; n = len(d)
    rat = [g[b].mean() / h[b].mean() for b in (np.random.choice(np.arange(n), n) for _ in range(2000))]
    out.append(dict(cond=name, model=d.model.iloc[0], intervention=d.intervention.iloc[0], n=n,
                    ratio=g.mean() / h.mean(), lo=np.percentile(rat, 2.5), hi=np.percentile(rat, 97.5),
                    cyc_ratio=d.cyc_model.mean() / d.cyc_human.mean()))
print(pd.DataFrame(out).round(3).to_string(index=False))
