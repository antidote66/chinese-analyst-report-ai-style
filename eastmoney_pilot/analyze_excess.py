#!/usr/bin/env python3
"""
超额词频分析（Kobak et al. 2025 方法移植到中文研报摘要）。

步骤:
 1. jieba 分词，每篇去重 -> 词的"篇频率" p_y(w) = 含该词的篇数 / 该年篇数
 2. 基线 2018–2022，对每个词做线性趋势外推到 2023–2026，得反事实频率 q_y(w)
 3. 超额频率 δ = p − q，超额比 r = p / q
 4. 输出候选超额词表（按 δ 和 r 两种排序），供人工三分类

输出:
  data/tokens.parquet          每篇的去重词集合
  data/word_year_freq.parquet  词×年 篇频率矩阵
  data/excess_words_{year}.csv 各年候选超额词（前 400）
"""
import re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import jieba

BASE_YEARS = [2018, 2019, 2020, 2021, 2022]
TEST_YEARS = [2023, 2024, 2025, 2026]
HAN = re.compile(r"^[一-鿿]{2,}$")   # 只保留 ≥2 个汉字的词
MIN_P = 0.002                                  # 测试年篇频率下限（0.2%）


DISC = re.compile(r"(免责|声明|请务必|务必阅读|参阅|敬请|咨询业务资格|执业证书|证书编号|研究所$|评级说明|分析师承诺|正文之后|最后一页|投资评级|信息披露)")


def clean(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not DISC.search(l))


def tokenize(text: str):
    return sorted({t for t in jieba.cut(clean(text)) if HAN.match(t)})


def main():
    d = pd.read_parquet("data/abstracts.parquet", columns=["infoCode", "abstract", "publishDate", "qType", "orgSName"])
    d = d[d.abstract.str.len() > 50].copy()
    d["year"] = d.publishDate.dt.year
    print(f"有效摘要 {len(d)} 篇")

    tok_path = Path("data/tokens.parquet")
    if tok_path.exists():
        tok = pd.read_parquet(tok_path)
    else:
        jieba.setLogLevel(60)
        d["tokens"] = [tokenize(t) for t in d.abstract]
        tok = d[["infoCode", "year", "qType", "orgSName", "tokens"]]
        tok.to_parquet(tok_path, index=False)
    print("分词完成")

    # 词×年 篇频率
    n_year = tok.groupby("year").size()
    ex = tok[["year", "tokens"]].explode("tokens").dropna()
    cnt = ex.groupby(["tokens", "year"]).size().unstack(fill_value=0)
    freq = cnt.div(n_year, axis=1)
    freq = freq[(freq[TEST_YEARS].max(axis=1) >= MIN_P) | (freq[BASE_YEARS].max(axis=1) >= MIN_P)]
    freq.to_parquet("data/word_year_freq.parquet")
    print(f"词表 {len(freq)} 个（篇频率≥{MIN_P:.1%}）")

    # 线性外推反事实
    X = np.array(BASE_YEARS, dtype=float)
    Y = freq[BASE_YEARS].values
    slope = ((X - X.mean()) * (Y - Y.mean(axis=1, keepdims=True))).sum(axis=1) / ((X - X.mean()) ** 2).sum()
    inter = Y.mean(axis=1) - slope * X.mean()
    rows = []
    base_mean = Y.mean(axis=1)
    base_min = np.maximum(Y.min(axis=1), 1.0 / n_year[BASE_YEARS].sum())
    for y in TEST_YEARS:
        q_trend = np.maximum(inter + slope * y, base_min)      # 趋势外推，但不低于基线最低年
        q = q_trend
        p = freq[y].values
        df = pd.DataFrame({"word": freq.index, "p": p, "q": q, "base_mean": base_mean,
                           "delta": p - q, "ratio": p / q,
                           "delta_mean": p - base_mean, "ratio_mean": p / np.maximum(base_mean, base_min), "year": y})
        df = df[df.p >= MIN_P]
        rows.append(df)
        top_d = df.sort_values("delta", ascending=False).head(400)
        top_r = df[df.p >= 0.005].sort_values("ratio", ascending=False).head(400)
        out = pd.concat([top_d.assign(rank_by="delta"), top_r.assign(rank_by="ratio")]).drop_duplicates("word")
        out = out.sort_values("delta", ascending=False)
        out.round(5).to_csv(f"data/excess_words_{y}.csv", index=False)
        print(f"\n== {y} == 篇数 {n_year[y]}")
        print("按超额频率 δ 前 40：")
        print(", ".join(f"{w}({p:.1%}←{q:.1%})" for w, p, q in top_d[["word", "p", "q"]].head(40).values))
        print("按超额比 r 前 40（p≥0.5%）：")
        print(", ".join(f"{w}({r:.1f}x)" for w, r in top_r[["word", "ratio"]].head(40).values))
    pd.concat(rows).to_parquet("data/excess_all.parquet", index=False)


if __name__ == "__main__":
    main()
