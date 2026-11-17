# Binoculars zero-shot detector — laptop run

Run on a Windows laptop (Ryzen 7 H 255, 32 GB RAM, CPU only), Nov 2026.

| File | Contents |
| --- | --- |
| `run_all.bat` | One-click pipeline: venv → crawl (`../../sina_corpus/sina_abstracts.py`) → score → summarize. Resumable. Parameters at the top (`YEARS`, `PER_YEAR`, `OBS_MODEL`, `PERF_MODEL`) |
| `zeroshot_detect.py` | Binoculars score (observer `Qwen/Qwen2.5-1.5B`, performer `Qwen/Qwen2.5-1.5B-Instruct`) plus GLTR-style rank stats; first 512 tokens per report |
| `summarize.py` | Per-year mean / sd / median, Cohen's d and Mann–Whitney p (last year vs first) |
| `out/zeroshot_scores.csv` | 1,996 scored reports (≈333 per year; 2019/2021/2023/2024/2025/2026, Oct 1–Nov 4 window). Columns: `rptid, year, date, org, rtype, ntok, logppl, xppl, binoculars, mean_rank, top10` |
| `out/summary.txt`, `out/zeroshot_summary.csv` | Output of `summarize.py` |

Note: `run_all.bat` expects `sina_abstracts.py` in the same folder (that is how it ran on the laptop);
copy it from `sina_corpus/` before running.

Main result (paper §5.4): Binoculars scores *rise* with year (2026 vs 2019: d = +0.55, p < 0.001),
i.e. later reports look *more* human to the detector. The step is at 2024, matching the Qwen2.5
training-data cutoff rather than the 2025 spread of DeepSeek, and it comes from strategy / macro /
industry / bond reports while company reports stay flat. We read this as training-cutoff confounding,
not as evidence about AI adoption.
