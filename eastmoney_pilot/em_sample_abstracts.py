#!/usr/bin/env python3
"""
分层抽样 + 下载PDF + 抽取首页“投资要点/核心观点”文本。

用法:
    python em_sample_abstracts.py --meta data/meta.parquet --out data/abstracts.parquet \
        --per-quarter 1000 --pdf-dir data/pdf --keep-pdf 0

分层: 年×季度×qType。每季度配额 --per-quarter，按 stock:industry:strategy+macro = 6:3:1 分配。
抽样固定随机种子(2026)，可复现。断点续传：已处理的 infoCode 写在输出文件里，重跑跳过。
只保留首页文本(raw_page1)和切出的摘要(abstract)，PDF 默认下完即删。
"""
import argparse, re, time, random, sys, io
from pathlib import Path
import requests
import pandas as pd
import pdfplumber

PDF_URL = "https://pdf.dfcfw.com/pdf/H3_{code}_1.pdf"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Referer": "https://data.eastmoney.com/report/",
    "Accept": "application/pdf,*/*",
}
SHARE = {0: 0.6, 1: 0.3, 2: 0.1}

# 摘要段落的起止标记（券商首页常见写法）
START_PAT = re.compile(r"(投资要点|核心观点|报告摘要|主要观点|核心结论|摘要|事件|投资摘要|事项|要点)[:：]?\s*$|^(投资要点|核心观点|核心结论|报告摘要)")
END_PAT = re.compile(r"^(风险提示|风险因素|风险|盈利预测与估值|盈利预测|投资建议|相关研究|作者|分析师|证券分析师|联系人|请务必阅读|免责声明|重要声明|财务摘要|盈利预测及估值)")


def stratified_sample(meta: pd.DataFrame, per_quarter: int, seed: int = 2026):
    m = meta.dropna(subset=["publishDate"]).copy()
    m["yq"] = m.publishDate.dt.year.astype(str) + "Q" + m.publishDate.dt.quarter.astype(str)
    picks = []
    rng = random.Random(seed)
    for yq, g in m.groupby("yq"):
        for q, share in SHARE.items():
            pool = g[g.qType == q]
            n = min(len(pool), int(round(per_quarter * share)))
            if n:
                picks.append(pool.sample(n=n, random_state=rng.randint(0, 10**9)))
    return pd.concat(picks, ignore_index=True)


BULLET_PAT = re.compile(r"^[■●◆•➤❖∙►►]")
META_PAT = re.compile(r"(执业证书|证书编号|分析师|联系人|@|电话|Tel|数据来源|资料来源|股价表现|市场数据|基本数据|相关研究|Table_|^\d{4}[年/-]\d{1,2}|^[\d.,%()（）\-\s]+$)")


def main_column_text(page) -> str:
    """按字符密度找出正文主栏（左起点众数 + 向右直到密度空档），裁剪后再抽文本，避免与侧栏行混排。"""
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    if not words:
        return page.extract_text() or ""
    W = float(page.width)
    # 1) 正文左边界 L：按字符数加权的 x0 众数
    bins = {}
    for w in words:
        n = len(w["text"]); bins[int(w["x0"] // 12)] = bins.get(int(w["x0"] // 12), 0) + n
    L = max(bins, key=bins.get) * 12
    # 2) 右边界 R：从 L 向右扫描 6pt 条带的字符密度，遇到低密度空档即停
    strip = 6
    dens = [0] * (int(W // strip) + 2)
    for w in words:
        if w["x0"] < L - 15:
            continue
        a, b = int(w["x0"] // strip), int(w["x1"] // strip)
        per = len(w["text"]) / max(1, b - a + 1)
        for s in range(a, min(b, len(dens) - 1) + 1):
            dens[s] += per
    peak = max(dens) if dens else 1
    R = W
    i = int(L // strip) + 10          # 至少先跨过 60pt
    while i < len(dens):
        if dens[i] < 0.08 * peak and dens[min(i + 1, len(dens) - 1)] < 0.08 * peak:
            R = i * strip; break
        i += 1
    if R - L < 0.35 * W:               # 裁太窄说明判断失败，退回整页
        return page.extract_text() or ""
    try:
        return page.crop((max(0, L - 6), 0, min(W, R + 6), float(page.height))).extract_text() or ""
    except Exception:
        return page.extract_text() or ""


def extract_abstract(text: str) -> str:
    lines = [l.strip() for l in text.splitlines()]
    start = None
    for i, l in enumerate(lines):
        if START_PAT.search(l):
            start = i + 1; break
    if start is None:                     # 无“投资要点”标题：从第一个项目符号行开始
        for i, l in enumerate(lines):
            if BULLET_PAT.search(l) and len(l) > 8:
                start = i; break
    if start is None:                     # 再退：跳过前几行标题区，取其余正文
        start = min(4, len(lines))
    buf = []
    for l in lines[start:]:
        if not l or META_PAT.search(l):
            continue
        if sum(c.isdigit() for c in l) > 0.4 * len(l):   # 财务表格行
            continue
        if END_PAT.search(l):
            break
        buf.append(l)
    return "\n".join(buf)


def fetch_pdf(session, code, tries=4):
    url = PDF_URL.format(code=code)
    for t in range(tries):
        try:
            r = session.get(url, headers=HEADERS, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                return None
            return r.content
        except Exception as e:  # noqa
            time.sleep(2 ** t + random.random())
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/meta.parquet")
    ap.add_argument("--out", default="data/abstracts.parquet")
    ap.add_argument("--per-quarter", type=int, default=1000)
    ap.add_argument("--pdf-dir", default="data/pdf")
    ap.add_argument("--keep-pdf", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.8)
    ap.add_argument("--limit", type=int, default=0, help="调试用：只处理前N份")
    args = ap.parse_args()

    meta = pd.read_parquet(args.meta)
    sample = stratified_sample(meta, args.per_quarter)
    print(f"抽样 {len(sample)} 份 / 元数据 {len(meta)} 条")
    print(sample.groupby(["yq", "qTypeName"]).size().unstack(fill_value=0).to_string())

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pdf_dir = Path(args.pdf_dir); pdf_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = set(pd.read_parquet(out, columns=["infoCode"]).infoCode)
    todo = sample[~sample.infoCode.isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"待处理 {len(todo)} 份（已完成 {len(done)}）")

    session = requests.Session()
    batch, n_ok, n_fail = [], 0, 0
    for i, row in enumerate(todo.itertuples(index=False), 1):
        code = row.infoCode
        content = fetch_pdf(session, code)
        rec = {"infoCode": code, "pdf_ok": content is not None, "raw_page1": "", "abstract": "", "n_pages": None}
        if content:
            try:
                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    rec["n_pages"] = len(pdf.pages)
                    page1 = main_column_text(pdf.pages[0])
                    if len(page1) < 200 and len(pdf.pages) > 1:  # 封面页，再取第二页
                        page1 += "\n" + main_column_text(pdf.pages[1])
                rec["raw_page1"] = page1
                rec["abstract"] = extract_abstract(page1)
                n_ok += 1
            except Exception as e:  # noqa
                print(f"  parse fail {code}: {e}", file=sys.stderr); n_fail += 1
            if args.keep_pdf:
                (pdf_dir / f"{code}.pdf").write_bytes(content)
        else:
            n_fail += 1
        batch.append(rec)
        if len(batch) >= 200 or i == len(todo):
            df = pd.DataFrame(batch)
            if out.exists():
                df = pd.concat([pd.read_parquet(out), df], ignore_index=True).drop_duplicates("infoCode")
            df.to_parquet(out, index=False)
            batch = []
            print(f"[{i}/{len(todo)}] ok={n_ok} fail={n_fail} abstract_hit={int((df.abstract.str.len()>50).sum())}")
        time.sleep(args.sleep)

    df = pd.read_parquet(out).merge(meta, on="infoCode", how="left")
    df.to_parquet(out, index=False)
    hit = (df.abstract.str.len() > 50).mean()
    print(f"\n完成。{len(df)} 份，摘要切出率 {hit:.1%}。切出率低于80%时把 raw_page1 发给我，我调 START/END 正则。")


if __name__ == "__main__":
    main()
