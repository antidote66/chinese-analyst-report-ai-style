#!/usr/bin/env python3
"""
按 infoCode 列表把东方财富研报 PDF 完整下载到本地长期保存（Mac mini 用）。

用法:
    # 只下抽样的 3.6 万份（约 50GB）
    python em_download_pdfs.py --list data/sample_codes.csv --pdf-dir ~/em_pdf

    # 下全量 40 万份（约 550GB，需要外接硬盘）
    python em_download_pdfs.py --list data/meta.parquet --pdf-dir /Volumes/外接盘/em_pdf

断点续传：已存在且以 %PDF 开头的文件直接跳过，随时 Ctrl-C 再重跑。
文件名: {pdf-dir}/{年}/{infoCode}.pdf
"""
import argparse, time, random, sys
from pathlib import Path
import requests
import pandas as pd

PDF_URL = "https://pdf.dfcfw.com/pdf/H3_{code}_1.pdf"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Referer": "https://data.eastmoney.com/report/",
    "Accept": "application/pdf,*/*",
}


def fetch(session, code, tries=4):
    for t in range(tries):
        try:
            r = session.get(PDF_URL.format(code=code), headers=HEADERS, timeout=90)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content if r.content.startswith(b"%PDF") else None
        except Exception:
            time.sleep(2 ** t + random.random())
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="csv 或 parquet，需含 infoCode、publishDate 列")
    ap.add_argument("--pdf-dir", required=True)
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()

    df = pd.read_parquet(args.list) if args.list.endswith(".parquet") else pd.read_csv(args.list, dtype=str)
    df["year"] = pd.to_datetime(df.publishDate, errors="coerce").dt.year.fillna(0).astype(int).astype(str)
    root = Path(args.pdf_dir).expanduser()
    session = requests.Session()
    n_ok = n_skip = n_fail = 0
    for i, row in enumerate(df.itertuples(index=False), 1):
        p = root / row.year / f"{row.infoCode}.pdf"
        if p.exists() and p.stat().st_size > 1000:
            n_skip += 1
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        c = fetch(session, row.infoCode)
        if c:
            p.write_bytes(c); n_ok += 1
        else:
            n_fail += 1
        if i % 200 == 0:
            print(f"[{i}/{len(df)}] 新下载 {n_ok} 跳过 {n_skip} 失败 {n_fail}", flush=True)
        time.sleep(args.sleep)
    print(f"完成: 新下载 {n_ok} 跳过 {n_skip} 失败 {n_fail}")


if __name__ == "__main__":
    main()
