# Chinese Financial Writing in the Age of Generative AI

**生成式 AI 时代的中文金融写作：语言变化、模型指纹与检测方法的局限**
Linguistic Change, Model Fingerprints, and the Limits of Detection

Andy Yu (虞慧晖), independent researcher, Shanghai — yhhzju@gmail.com

This repository holds the paper, the code and the derived data tables for a replication of the
excess-vocabulary method (Kobak et al. 2024) on Chinese sell-side analyst reports, 2019–2026,
together with paired controlled experiments on DeepSeek and Qwen and an evaluation of AI-text
detectors on this genre.

本仓库是论文《生成式 AI 时代的中文金融写作》的代码与数据清单：把 Kobak 等的超额词频方法复现到
2019–2026 年中文券商研报，用 DeepSeek 与千问的逐篇配对受控对照检验归因，并评估 AI 文本检测器在研报文体上的表现。

## Paper / 论文

- `paper/paper_en_v1.pdf` — English full text.
- `paper/paper_zh_v1.pdf` — Chinese full text (English title and abstract on p. 2).

## Layout / 目录

| Path | Contents |
| --- | --- |
| `sina_corpus/` | Crawler for the Sina Finance research-report archive (main corpus), window and monthly-panel sampling, report-ID lists for every sample used in the paper |
| `experiments/` | Word-table construction and manual labels (151 style / 125 event / 24 format words), monthly panel and break-point regression, HC3 and Binoculars scoring, self-trained RoBERTa detector, paired DeepSeek / Qwen controlled experiments (prompts and outputs), Appendix A consequence tests |
| `eastmoney_pilot/` | Pilot on the Eastmoney (东方财富) free report API, run before the corpus was switched to Sina: metadata fetch (402,244 reports, 2018–2026), stratified PDF sampling (36,000 reports) and the first excess-word pass. Not used in the paper's reported results; kept for comparison |

## Reproduction / 复现

1. Main corpus: run `sina_corpus/sina_abstracts.py` (resumable; outputs `data/sina_abstracts/{year}.jsonl`).
   Summaries are the archive's own text; they are not redistributed here — only report-ID lists and scripts.
2. Word table: `experiments/` scripts compute document frequencies (jieba, words of ≥2 Chinese characters),
   excess frequency δ and ratio r against the 2019 same-window baseline, and apply the frozen 151-word list.
3. Monthly panel: piecewise linear regression with slope breaks at 2023-01 and 2025-01, Newey–West errors,
   grid search for the second break between 2023-07 and 2025-12.
4. Detectors: HC3 Chinese RoBERTa (`Hello-SimpleAI/chatgpt-detector-roberta-chinese`), Binoculars with
   Qwen2.5-0.5B/1.5B (+Instruct), and the self-trained `hfl/chinese-roberta-wwm-ext` classifier
   (positives: 2,359 DeepSeek texts; negatives: 2,359 human reports from 2019; the paper says 2,400, see `TODO.md`).
5. Controlled experiments: DeepSeek (`deepseek-flash`) and Qwen (`qwen-plus`), temperature 0.7, three
   interventions (light polish, deep rewrite, generation from the fact skeleton), paired to the 2019 originals.

All computation ran on one Mac mini, one laptop and one cloud container; no GPU cluster is needed.

## Status / 状态

As of 2026-11-18 the code, word lists, report-ID lists for all three samples (23,563 / 94,201 / 59,070),
detector data splits and metrics, and the controlled-experiment prompts and multiplier computation are in the
repository. Sample counts and the main numbers (eight-year index, break-point regression, §5.5 multipliers and
CIs, detector test metrics) were re-run from the saved files and match the paper; see `sina_corpus/README.md`
and `experiments/README.md`. Open items (polish/rewrite model outputs, three count corrections in the paper) are
listed in `TODO.md`. The `eastmoney_pilot/` folder is complete.

截至 2026-11-18，代码、词表、三套样本的编号清单、检测器数据划分与指标、受控对照的提示词与倍数计算已全部入库，
样本数与主要数值已从保存的文件重跑核对、与论文一致。未决事项见 `TODO.md`。

## Citation

```
Yu, H. (2026). Chinese Financial Writing in the Age of Generative AI: Linguistic Change,
Model Fingerprints, and the Limits of Detection. arXiv preprint.
```

## License

Code: MIT. Paper text and derived tables: CC BY 4.0. Research-report summaries belong to their
publishers and are not redistributed.
