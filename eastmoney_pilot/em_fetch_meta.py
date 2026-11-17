#!/usr/bin/env python3
"""
东方财富研报元数据全量拉取（免费公开接口，无需登录）。

用法:
    python em_fetch_meta.py --start 2018-01 --end 2026-11 --out data/meta.parquet

按月分段、按报告类型(qType 0-4)分页拉取 reportapi.eastmoney.com/report/list，
断点续传：已完成的 (月份, qType) 记录在 data/meta_progress.json，重跑自动跳过。
"""
import argparse, json, time, random, sys
from pathlib import Path
from datetime import date
import requests
import pandas as pd

API = "https://reportapi.eastmoney.com/report/list"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Referer": "https://data.eastmoney.com/report/",
    "Accept": "application/json, text/plain, */*",
}
QTYPES = {0: "stock", 1: "industry", 2: "strategy", 3: "macro", 4: "morning"}
KEEP = ["infoCode", "title", "stockName", "stockCode", "orgCode", "orgSName",
        "publishDate", "industryCode", "industryName", "emIndustryCode",
        "emRatingName", "lastEmRatingName", "ratingChange", "reportType", "column",
        "researcher", "authorID", "attachPages", "attachSize",
        "predictThisYearEps", "predictNextYearEps", "predictThisYearPe", "predictNextYearPe",
        "indvAimPriceT", "indvAimPriceL", "count"]


def month_ranges(start: str, end: str):
    y, m = map(int, start.split("-"))
    ye, me = map(int, end.split("-"))
    while (y, m) <= (ye, me):
        first = date(y, m, 1)
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        last = date(ny, nm, 1) - pd.Timedelta(days=1)
        yield f"{y}-{m:02d}", first.isoformat(), last.isoformat()
        y, m = ny, nm


def fetch_page(session, qtype, begin, end, page, size=100, tries=5):
    params = {"qType": qtype, "pageSize": size, "pageNo": page,
              "beginTime": begin, "endTime": end, "industryCode": "*",
              "fields": "", "orgCode": "", "rcode": ""}
    for t in range(tries):
        try:
            r = session.get(API, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa
            wait = 2 ** t + random.random()
            print(f"  retry {t+1}/{tries} qType={qtype} {begin} p{page}: {e}; sleep {wait:.1f}s", file=sys.stderr)
            time.sleep(wait)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01")
    ap.add_argument("--end", default=date.today().strftime("%Y-%m"))
    ap.add_argument("--out", default="data/meta.parquet")
    ap.add_argument("--qtypes", default="0,1,2")  # qType 3(宏观) 接口实测返回空，暂不拉
    ap.add_argument("--sleep", type=float, default=0.6, help="每页之间的间隔秒数，别太快")
    args = ap.parse_args()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    prog_path = out.parent / "meta_progress.json"
    progress = json.loads(prog_path.read_text()) if prog_path.exists() else {}
    parts_dir = out.parent / "meta_parts"; parts_dir.mkdir(exist_ok=True)

    session = requests.Session()
    qtypes = [int(q) for q in args.qtypes.split(",")]

    for ym, begin, end in month_ranges(args.start, args.end):
        for q in qtypes:
            key = f"{ym}:{q}"
            if progress.get(key) == "done":
                continue
            rows, page = [], 1
            while True:
                js = fetch_page(session, q, begin, end, page)
                if js is None:
                    print(f"GIVE UP {key} page {page}", file=sys.stderr); break
                data = js.get("data") or []
                for d in data:
                    d = {k: d.get(k) for k in KEEP}
                    if isinstance(d.get("authorID"), list):
                        d["authorID"] = ",".join(d["authorID"])
                    # 接口返回类型不稳定（"" / int 混用），统一存为字符串，分析时再转
                    d = {k: ("" if v is None else str(v)) for k, v in d.items()}
                    d["qType"] = q; d["qTypeName"] = QTYPES[q]
                    rows.append(d)
                total_pages = int(js.get("TotalPage") or 1)
                if page >= total_pages or not data:
                    break
                page += 1
                time.sleep(args.sleep)
            if rows:
                pd.DataFrame(rows).to_parquet(parts_dir / f"{ym}_{q}.parquet", index=False)
            progress[key] = "done"
            prog_path.write_text(json.dumps(progress, ensure_ascii=False, indent=0))
            print(f"{key}: {len(rows)} rows, hits={js.get('hits') if js else '?'}")
            time.sleep(args.sleep)

    parts = sorted(parts_dir.glob("*.parquet"))
    if parts:
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        df = df.drop_duplicates("infoCode")
        df["publishDate"] = pd.to_datetime(df["publishDate"], errors="coerce")
        df.to_parquet(out, index=False)
        print(f"\n合并完成: {len(df)} 条 -> {out}")
        print(df.groupby([df.publishDate.dt.year, "qTypeName"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
