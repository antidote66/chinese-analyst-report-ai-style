#!/usr/bin/env python3
"""
断点曲线：同一日历窗口（10-01 至 11-04）逐年的词组文档频率。
输入：若干 jsonl 目录；只保留 date 在每年 10-01..11-04 的文档。
输出：out_curves/year_curves.csv、year_curves.png、style_vs_cycle.png
"""
import json, re, sys, collections
from pathlib import Path
import pandas as pd, jieba

import os
WLO = os.environ.get("WLO","10-01")
CJK = re.compile(r"^[一-鿿]{2,}$")
lab = pd.read_csv("out_pilot/candidate_words_labeled_v1.csv")
S = lab[lab.label.str.startswith("S")].word.tolist()
E = lab[lab.label.str.startswith("E")].word.tolist()
CYC = "下滑 下降 亏损 减少 放缓 疲软 低迷 恶化 收缩 回落 转亏".split()
FOCUS = "承压 修复 扰动 节奏 兑现 韧性 验证 协同 叠加 赋能 深耕".split()
ALL = set(S) | set(E) | set(CYC) | set(FOCUS)

docs, seen = [], set()
for p in sys.argv[1:]:
    for f in sorted(Path(p).glob("*.jsonl")):
        for line in f.open(encoding="utf-8"):
            try: d = json.loads(line)
            except Exception: continue
            if d["rptid"] in seen or not d.get("text") or len(d["text"]) < 80: continue
            md = d["date"][5:]
            if not (WLO <= md <= "11-04"): continue
            seen.add(d["rptid"]); docs.append(d)

cnt, n = collections.defaultdict(collections.Counter), collections.Counter()
for d in docs:
    y = d["date"][:4]; n[y] += 1
    cnt[y].update({w for w in jieba.cut(d["text"]) if CJK.match(w)} & ALL)
years = sorted(n)
print("窗口内各年篇数:", dict(n), file=sys.stderr)

rows = []
for y in years:
    g = lambda ws: sum(cnt[y][w] for w in ws) / (n[y] * len(ws))
    r = {"year": y, "n": n[y], "S_style": g(S), "E_event": g(E), "CYC_cycle": g(CYC)}
    for w in FOCUS: r[w] = cnt[y][w] / n[y]
    rows.append(r)
df = pd.DataFrame(rows); out = Path("out_curves"); out.mkdir(exist_ok=True)
df.to_csv(out / "year_curves.csv", index=False, encoding="utf-8-sig")
print(df.round(3).to_string(index=False))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if "NotoSansCJK" in f:
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name(); break
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for k, lbl in [("S_style", "功能/风格词组"), ("CYC_cycle", "周期词组"), ("E_event", "事件词组")]:
    ax[0].plot(df.year, df[k] / df[k].iloc[0], marker="o", label=lbl)
for bp, txt in [("2023", "ChatGPT 2022-11"), ("2025", "DeepSeek-R1 2025-01")]:
    if bp in years:
        x = years.index(bp) - 0.5
        ax[0].axvline(x, ls="--", c="gray"); ax[0].text(x, ax[0].get_ylim()[1] if False else 1.0, txt, fontsize=7, rotation=90, va="bottom", ha="right", color="gray")
ax[0].set_title("各词组文档频率（以首年=1），每年 " + WLO + "~11-04 窗口"); ax[0].legend()
for w in FOCUS:
    ax[1].plot(df.year, df[w], marker="o", label=w)
ax[1].set_title("候选风格词逐年文档频率"); ax[1].legend(ncol=2, fontsize=8)
for a in ax: a.tick_params(axis="x", rotation=45)
fig.tight_layout(); fig.savefig(out / "year_curves.png", dpi=150)
print("saved", out / "year_curves.png", file=sys.stderr)
