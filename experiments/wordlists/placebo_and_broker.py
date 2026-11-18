#!/usr/bin/env python3
"""
识别检验 1：安慰剂对比 —— 功能词（不含经济信息）vs 周期词（含经济信息）vs 事件词 的文档频率变化。
识别检验 2：横截面 —— 同一时间窗口内，各券商的“风格词”平均文档频率（初版 AI 味指数）。
输入：目标期/基线期 jsonl 目录；输出：markdown 报告。
"""
import json, sys, collections, re
from pathlib import Path
import jieba

# 词组由我（Claude）初标，Andy 抽查。原则：
#   功能词 = 不指向任何经济事实，只改变“说法”的词；周期词 = 直接描述经济/利润方向的词；事件词 = 2026 年才有的实体/事件。
GROUPS = {
    "功能词(风格)": "承压 修复 扰动 叠加 节奏 延续 释放 支撑 布局 推进 拓展 落地 优化 协同 增量 放量 后续 对应 形成 有望 持续 值得 关注 综上 整体 边际 弹性 兑现 验证 催化 向好 改善 韧性".split(),
    "周期词(经济内容)": "下滑 下降 亏损 减少 放缓 疲软 低迷 恶化 收缩 回落 转亏 增长 提升 上升 扩张 改善 复苏".split(),
    "事件词(2026实体)": "算力 液冷 人形 具身 推理 端侧 智算 出海 地缘 中东 美伊 关税 反内卷 新质 北交所".split(),
    "格式词(研报模板)": "扣非 毛利率 费用率 净利率 截至 季报 中报 年报 同比 环比 归母".split(),
}

CJK = re.compile(r"^[一-鿿]{2,}$")

def load(paths, years):
    docs = []
    seen = set()
    for p in paths:
        for f in sorted(Path(p).glob("*.jsonl")):
            for line in f.open(encoding="utf-8"):
                try: d = json.loads(line)
                except Exception: continue
                if d["rptid"] in seen or not d.get("text") or len(d["text"]) < 80: continue
                if d["date"][:4] not in years: continue
                seen.add(d["rptid"]); docs.append(d)
    return docs

def docfreq(docs, words):
    cnt = collections.Counter();
    for d in docs:
        toks = {w for w in jieba.cut(d["text"]) if CJK.match(w)}
        cnt.update(toks & words)
    return cnt

def main():
    T = load(["mac", "data"], {"2026"}); B = load(["base"], {"2019", "2018"})
    allw = {w for g in GROUPS.values() for w in g}
    cT, cB = docfreq(T, allw), docfreq(B, allw)
    nT, nB = len(T), len(B)
    out = [f"# 安慰剂检验：2026-10/11（{nT} 篇） vs 2019-10/11（{nB} 篇）\n",
           "各组取组内词文档频率的平均值；若 AI 味真实上升，功能词组涨幅应显著高于周期词组。\n"]
    for g, ws in GROUPS.items():
        rows = [(w, cT[w]/nT, cB[w]/nB) for w in ws]
        mT = sum(r[1] for r in rows)/len(rows); mB = sum(r[2] for r in rows)/len(rows)
        out.append(f"\n## {g}  组均文档频率 {mB:.3f} → {mT:.3f}  （×{mT/max(mB,1e-6):.2f}）\n")
        out.append("| 词 | 2019 | 2026 | 倍数 |\n|---|---|---|---|")
        for w, a, b in sorted(rows, key=lambda r: -(r[1]/max(r[2],1e-4))):
            out.append(f"| {w} | {b:.3f} | {a:.3f} | {a/max(b,1e-4):.1f} |")
    # 横截面：券商 AI 味指数（功能词组平均文档频率），只看 2026，≥80 篇的券商
    style = set(GROUPS["功能词(风格)"])
    by_org = collections.defaultdict(list)
    for d in T: by_org[d["org"]].append(d)
    rows = []
    for org, ds in by_org.items():
        if len(ds) < 80: continue
        c = docfreq(ds, style)
        rows.append((org, len(ds), sum(c[w] for w in style)/(len(ds)*len(style))))
    rows.sort(key=lambda r: -r[2])
    out.append(f"\n# 横截面：各券商 2026-10/11 风格词指数（≥80 篇，共 {len(rows)} 家）\n")
    out.append("| 排名 | 券商 | 篇数 | 风格词指数 |\n|---|---|---|---|")
    for i, (org, n, s) in enumerate(rows, 1):
        out.append(f"| {i} | {org} | {n} | {s:.3f} |")
    if rows:
        out.append(f"\n最高/最低 = {rows[0][2]/rows[-1][2]:.2f} 倍；同一时间窗口内，宏观环境对所有券商相同，差异只能来自写作方式。")
    Path("out_pilot/placebo_broker.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))

if __name__ == "__main__":
    main()
