# Experiments

`results/` holds every numeric table reported in the paper, as CSV, so figures can be checked and re-plotted
without the raw corpus.

| File | Paper section |
| --- | --- |
| `annual_index_nov1-4.csv` | §5.2 table (eight-year same-window index) |
| `monthly_panel_annual_means.csv`, `piecewise_regression.csv` | §5.2 monthly panel and break-point test |
| `detector_hc3.csv`, `detector_binoculars.csv`, `detector_selftrained.csv` | §5.4 detectors |
| `controlled_comparison.csv`, `controlled_comparison_words.csv` | §5.5 paired DeepSeek / Qwen experiments |
| `appendixA_car60.csv` | Appendix A.2 |

`binoculars_laptop/` holds the complete, runnable code and per-report scores for the 1.5B Binoculars re-run (§5.4).

## Code status

The scripts for the Sina-corpus analyses (frozen 151-word index, monthly-panel regression, HC3 scoring, self-trained detector, paired generation with DeepSeek and Qwen) ran on the author's local machines and
in short-lived cloud workspaces. They are being collected and will be added here; until then they are available
from the author on request (yhhzju@gmail.com). The method is fully specified in §4 of the paper.

The Eastmoney pilot (`../eastmoney_pilot/`) is complete and runnable: it implements the same excess-vocabulary
pipeline (jieba document frequencies, counterfactual baseline, δ and r rankings) on a second corpus.
