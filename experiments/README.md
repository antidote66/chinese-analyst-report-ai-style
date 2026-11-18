# Experiments

Everything here comes from the machines and workspaces the paper was computed on (Mac mini, laptop, and the
2026-11-05/06 cloud workspaces whose data and code package was saved on 2026-11-07). Scripts are kept as they
were run; their absolute paths (`/home/claude/sina/…`, `/home/claude/final/…`, `/home/claude/gen/…`) refer to that
workspace layout. Report abstracts are not redistributed (see `../sina_corpus/`).

Every numeric check listed under **Verified** was re-run on 2026-11-18 from the saved files and matches the paper
exactly (to the printed precision).

| Folder | Paper | Contents | Verified |
| --- | --- | --- | --- |
| `wordlists/` | §4.1–4.2, §5.1, §5.3 | `candidate_words_labeled_v1.csv` (top-300 δ candidates with labels and p_T/p_B/δ/r), split files `style_words_S151.csv`, `event_words_E125.csv`, `format_words_F24.csv`, `cycle_words_C11.csv`; `ai_style_pipeline.py` (excess-vocabulary pipeline), `placebo_and_broker.py` (placebo grouping), `build_wordlist.py`, `style_word_lifecycle.csv`, `doc_freq_by_year_full.csv` | 151 / 125 / 24 / 11 words; yearly document frequencies reproduce from the 94,201-report window list |
| `index_panel/` | §4.3, §5.2 | `year_curves.py` (eight-year Nov 1–4 index; run as `WLO=11-01 python year_curves.py mac data base`), `panel_analysis.py` (monthly panel → `panel_monthly.csv`, report-level regression clustered by month), `panel_robust.py` (broker / report-type FE, list-order check, placebo knots), `piecewise_breaksearch.py` (79 equal-weighted months, Newey–West, break grid 2023-07…2025-12) | `piecewise_breaksearch.py` on `panel_monthly.csv` gives +0.00054 [0.00039, 0.00068], −0.00056 [−0.00096, −0.00016], +0.00219 [0.00167, 0.00271], best break 2025-04; Nov 1–4 index values for all eight years |
| `hc3/` | §4.5, §5.4 | `score_detector.py` (HC3 `Hello-SimpleAI/chatgpt-detector-roberta-chinese`, 1,100 per year, seed 7), per-report scores `detector_scores_strat.csv` (2019–2024) and `detector_scores_2026_partial.csv` (2026, first full pass before the run was stopped) | per-year means / flagged shares = `results/detector_hc3.csv` |
| `binoculars_laptop/` | §4.5, §5.4 | Binoculars 1.5B run on the laptop (code, per-report scores, summary) | — (already complete) |
| `selftrained_detector/` | §4.5, §5.4 | `train_detector.py` (hfl/chinese-roberta-wwm-ext, frozen embeddings + first 6 layers, 320 chars, 2 epochs), `data_split_by_rptid.csv` + `detector_split.py` (train/val/test by report ID), test metrics `detector_eval.json`, cross-model test `xmodel_eval2.py` / `xmodel_eval2.json`, yearly scoring `score_and_plot.py` → `detector_by_year.csv`, `scores_2026.csv`, `train.log` | test set n = 710 (226 generated, 131 polished, 353 human), AUC 0.956; 2026 flagged 18.5% |
| `controlled_comparison/` | §3.3, §4.6, §5.5 | Generation scripts with the prompts (`gen_samples.py` DeepSeek, `gen_qwen.py` Qwen, `gen_claude.py`, `regen.py`; API keys are read from local `.env.*` files that are **not** in the repo), `cross_model.py` (multipliers + bootstrap CI), `wordcorr.py` (Spearman word-level correlation), `paired_word_stats.csv` (per pair: S-group and cycle-group occurrence in model text and paired 2019 original), `multiplier_ci.py` (recomputes the §5.5 table from that file), `model_outputs/` (independent-generation texts, DeepSeek 1,500 and Qwen 800), `cross_model_results.csv`, `wordcorr.csv`; `balanced3.py` / `table_main.csv` are the later three-model, length-controlled robustness version | sample sizes 859 / 800 / 1,500 (DeepSeek) and 800 / 800 / 800 (Qwen); multipliers and CIs 1.013 (1.007–1.018), 1.134 (1.121–1.147), 1.043 (1.021–1.066), 1.789 (1.761–1.817), 2.534 (2.482–2.587), 1.640 (1.600–1.679); ρ values |
| `appendixA/` | Appendix A | `score_reports.py` (per-report style / euphemism / negative densities), `car_test.py` (5/20/60-day CAR regressions), `fetch_eps.py`, result tables `car_regressions.csv`, `convergence_by_year.csv`, `forecast_broker_year.csv`; `session_log_extracts/` holds the inline analysis code (industry-FE CAR, |CAR|, narrative convergence, EPS matching and forecast error, Kim-style abnormal trading value, EPS field audits) copied verbatim from the session log | tables match Appendix A; extracts were not re-run here (price and Eastmoney inputs are not in the repo) |
| `results/` | all | every numeric table reported in the paper, as CSV | — |

## What is not here, and why

- **Polish / deep-rewrite model outputs** (DeepSeek and Qwen): these texts are close rewrites of the brokers'
  2019 abstracts, so publishing them would in effect republish the originals. `paired_word_stats.csv` carries
  everything needed to recompute the §5.5 numbers. Available from the author for verification.
- **2019 fact skeletons** (`gen_skeletons_2019.json`): the "facts" are sentences cut from the original abstracts;
  the report IDs are in `paired_word_stats.csv` and `selftrained_detector/data_split_by_rptid.csv`.
- **Model checkpoint** of the self-trained detector (≈400 MB); retrain with `train_detector.py`.
- **API keys**: never committed; scripts read them from local `.env.deepseek`, `.env.qwen`, `.env.anthropic`.

## Environment

Python 3.10+, `jieba==0.42.1`, pandas, numpy, scipy, statsmodels, scikit-learn, requests, torch, transformers.
