#!/usr/bin/env python3
"""
中文研报摘要的 AI 化测量 —— Kobak 等 (2024) 超额词法 (excess vocabulary) 的中文实现。

方法要点（与 Kobak 2024, Sci. Adv. 一致）：
  * 单位是“文档频率”：一个词在一个摘要里出现 ≥1 次记 1，p = 含该词的文档数 / 文档总数。
  * 对目标期 T（如 2026）与基线期 B（如 2018–2019）：
        δ = p_T − p_B           （超额频率，看“量”）
        r = p_T / p_B           （频率比，看“倍数”）
  * Kobak 用的是基线期线性外推的反事实；研报语料跨度短，先用“基线期均值”做反事实，
    后续有 2018–2022 全量后改为线性外推（函数 counterfactual_linear 已留接口）。
  * 排除：纯数字/字母、单字、停用词；保留 2 字及以上中文词。

用法:
  python ai_style_pipeline.py --inputs dirA dirB ... --target-years 2026 --base-years 2018 2019 --out out/
输出:
  out/doc_freq_by_year.csv      每词每年文档频率
  out/excess_words_<T>.csv      超额词表（按 δ、r 排序，含 p_T, p_B, 文档数）
  out/monthly_index.csv/.png    按月 AI 味指数（top-K 风格词的平均文档频率）
"""
import argparse, json, re, sys, collections
from pathlib import Path
import pandas as pd
import jieba

STOP = set("""的 了 和 与 及 或 在 是 为 等 对 以 并 将 从 于 其 之 也 有 中 上 下 后 前 内 外 年 月 日 元 亿元 万元 公司 我们 同比 环比 预计 分别 实现 增长 下降 收入 利润 净利润 归母 营收 营业 业绩 报告 风险 提示 维持 评级 买入 增持 推荐 目标价 盈利 预测 投资 建议 行业 市场 本期 上期 季度 Q1 Q2 Q3 Q4 一季度 二季度 三季度 四季度 上半年 下半年 全年 同期 期间 亿 万 百分点 pct""".split())

CJK = re.compile(r"^[一-鿿]{2,}$")

def iter_docs(paths):
    for p in paths:
        p = Path(p)
        files = [p] if p.is_file() else sorted(p.glob("*.jsonl"))
        for f in files:
            with f.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    if d.get("text") and len(d["text"]) >= 80:
                        yield d

def tokenize(text):
    toks = set()
    for w in jieba.cut(text):
        w = w.strip()
        if CJK.match(w) and w not in STOP:
            toks.add(w)
    return toks

def build_doc_freq(docs, by="year"):
    """返回 {period: Counter(word->doc count)}, {period: n_docs}"""
    cnt, n = collections.defaultdict(collections.Counter), collections.Counter()
    seen = set()
    for d in docs:
        if d["rptid"] in seen:
            continue
        seen.add(d["rptid"])
        key = d["date"][:4] if by == "year" else d["date"][:7]
        n[key] += 1
        cnt[key].update(tokenize(d["text"]))
    return cnt, n

def excess(cnt, n, target_years, base_years, min_docs=20):
    """δ = p_T - p_B, r = p_T / p_B ；p_B 为基线各年文档频率的均值（文档加权）。"""
    def pool(years):
        c, tot = collections.Counter(), 0
        for y in years:
            if y in cnt:
                c.update(cnt[y]); tot += n[y]
        return c, tot
    cT, nT = pool(target_years); cB, nB = pool(base_years)
    rows = []
    for w, k in cT.items():
        if k < min_docs:
            continue
        pT = k / nT
        pB = cB.get(w, 0) / nB if nB else 0.0
        pB_s = max(pB, 0.5 / nB) if nB else 1e-6   # 零频率平滑，避免 r=∞
        rows.append((w, k, pT, pB, pT - pB, pT / pB_s))
    df = pd.DataFrame(rows, columns=["word", "docs_T", "p_T", "p_B", "delta", "ratio"])
    return df, nT, nB

def monthly_index(cnt_m, n_m, words):
    rows = []
    for m in sorted(cnt_m):
        if n_m[m] < 30:
            continue
        rows.append((m, n_m[m], sum(cnt_m[m][w] for w in words) / (n_m[m] * len(words))))
    return pd.DataFrame(rows, columns=["month", "n_docs", "index"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--target-years", nargs="+", required=True)
    ap.add_argument("--base-years", nargs="+", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--topk", type=int, default=40)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    docs = list(iter_docs(a.inputs))
    print(f"文档数 {len(docs)}", file=sys.stderr)
    cnt_y, n_y = build_doc_freq(docs, "year")
    cnt_m, n_m = build_doc_freq(docs, "month")
    print("各年文档数:", dict(sorted(n_y.items())), file=sys.stderr)

    # 逐年文档频率表（只保留在任一年 ≥ 1% 的词，控制体积）
    vocab = set()
    for y, c in cnt_y.items():
        vocab |= {w for w, k in c.items() if k / n_y[y] >= 0.01}
    pd.DataFrame({y: {w: cnt_y[y][w] / n_y[y] for w in vocab} for y in sorted(cnt_y)}).to_csv(out / "doc_freq_by_year.csv", encoding="utf-8-sig")

    df, nT, nB = excess(cnt_y, n_y, a.target_years, a.base_years)
    df.sort_values("delta", ascending=False).to_csv(out / f"excess_words_{'_'.join(a.target_years)}.csv", index=False, encoding="utf-8-sig")
    print(f"目标期文档 {nT}，基线期文档 {nB}", file=sys.stderr)

    # 两张榜：按 δ（量）和按 r（倍数，限 p_T ≥ 1%）
    top_delta = df.sort_values("delta", ascending=False).head(a.topk)
    top_ratio = df[df.p_T >= 0.01].sort_values("ratio", ascending=False).head(a.topk)
    with (out / "top_words.md").open("w", encoding="utf-8") as fh:
        fh.write(f"# 超额词表  目标期 {a.target_years}（{nT} 篇） vs 基线 {a.base_years}（{nB} 篇）\n\n")
        for title, t in [("按超额频率 δ 排序（绝对量）", top_delta), ("按频率比 r 排序（倍数，p_T≥1%）", top_ratio)]:
            fh.write(f"## {title}\n\n| 词 | p_T | p_B | δ | r |\n|---|---|---|---|---|\n")
            for _, r in t.iterrows():
                fh.write(f"| {r.word} | {r.p_T:.3f} | {r.p_B:.3f} | {r.delta:+.3f} | {r.ratio:.1f} |\n")
            fh.write("\n")

    # 月度指数：用 δ 榜前 topk 词（用户分类前先全用，分类后换成风格词子集）
    mi = monthly_index(cnt_m, n_m, list(top_delta.word))
    mi.to_csv(out / "monthly_index.csv", index=False)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for f in font_manager.findSystemFonts():
            if "NotoSansCJK" in f: matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name(); break
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(mi.month, mi["index"], marker="o")
        ax.set_title(f"研报摘要 AI 味指数（top{a.topk} 超额词平均文档频率）")
        ax.set_ylabel("平均文档频率"); ax.tick_params(axis="x", rotation=60, labelsize=7)
        fig.tight_layout(); fig.savefig(out / "monthly_index.png", dpi=150)
    except Exception as e:
        print("绘图跳过:", e, file=sys.stderr)
    print("done ->", out, file=sys.stderr)

if __name__ == "__main__":
    main()
