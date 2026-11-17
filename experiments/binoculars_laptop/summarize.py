#!/usr/bin/env python3
"""汇总 out/zeroshot_scores.csv：按年份的均值/标准差/中位数，以及 2026 vs 2019 的效应量。用法: python summarize.py out"""
import sys, os
import pandas as pd

out = sys.argv[1] if len(sys.argv) > 1 else "out"
d = pd.read_csv(os.path.join(out, "zeroshot_scores.csv"))
print("已打分", len(d), "\n")
cols = ["logppl", "xppl", "binoculars", "mean_rank", "top10"]
g = d.groupby("year")[cols].agg(["count", "mean", "std", "median"]).round(3)
print(g.to_string())
ys = sorted(d.year.unique())
if len(ys) >= 2:
    a, b = d[d.year == ys[0]], d[d.year == ys[-1]]
    print(f"\nCohen's d ({ys[-1]} vs {ys[0]}):")
    for c in cols:
        sp = ((a[c].var() + b[c].var()) / 2) ** 0.5
        print(f"  {c:11s} {((b[c].mean() - a[c].mean()) / sp) if sp else float('nan'):+.3f}")
try:
    from scipy.stats import mannwhitneyu
    if len(ys) >= 2:
        print(f"\nMann-Whitney p ({ys[-1]} vs {ys[0]}):")
        for c in cols:
            print(f"  {c:11s} p={mannwhitneyu(a[c], b[c]).pvalue:.4f}")
except ImportError:
    pass
g.to_csv(os.path.join(out, "zeroshot_summary.csv"), encoding="utf-8-sig")
print("\n已写", os.path.join(out, "zeroshot_summary.csv"))
