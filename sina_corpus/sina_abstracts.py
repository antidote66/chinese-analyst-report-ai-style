#!/usr/bin/env python3
"""
抓取新浪财经研报库的研报【摘要全文】（HTML 文本，不下载 PDF），按发布日期遍历。
用法: python3 sina_abstracts.py --start 2018-01-01 --end 2026-11-04 --out data/sina_abstracts
输出: {out}/{年}.jsonl，每行一条 {rptid,title,rtype,date,org,researcher,symbol,stock,text}
断点续传: 按日期记录在 {out}/done_dates.txt；已抓的 rptid 跳过。
"""
import argparse, re, json, time, random, html, sys
from pathlib import Path
from datetime import date, timedelta
import requests

UA = {"User-Agent": "Mozilla/5.0"}
LIST = "https://stock.finance.sina.com.cn/stock/go.php/vReport_List/kind/search/index.phtml?t1=6&pubdate={d}&p={p}"
ROW = re.compile(r'<td>\d+</td>\s*<td class="tal f14">\s*<a[^>]*title="([^"]*)"[^>]*href="([^"]*rptid/(\d+)/[^"]*)"[^>]*>.*?</a>\s*</td>\s*<td>([^<]*)</td>\s*<td>(\d{4}-\d{2}-\d{2})</td>\s*<td>(.*?)</td>\s*<td>(.*?)</td>', re.S)

def get(s, url, tries=4):
    for t in range(tries):
        try:
            r = s.get(url, headers=UA, timeout=60); r.raise_for_status()
            r.encoding = "gb18030"; return r.text
        except Exception:
            time.sleep(2 ** t + random.random())
    return None

def parse_show(t):
    m = re.search(r'class="blk_container">(.*?)</div>', t, re.S)
    txt = html.unescape(re.sub(r'<[^>]+>', '', m.group(1))).strip() if m else ""
    sym = re.search(r'vReport_List/kind/search/index\.phtml\?t1=2&symbol=([^"&]+)', t)
    stock = re.search(r'<span class="f_blue">[^<]*</span>\s*(?:<a[^>]*>)?([^<（(]+)', t)
    return re.sub(r'[ \t　]+', ' ', txt), (sym.group(1) if sym else ""), ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--out", default="data/sina_abstracts"); ap.add_argument("--sleep", type=float, default=0.4); ap.add_argument("--cap", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    donef = out / "done_dates.txt"; done = set(donef.read_text().split()) if donef.exists() else set()
    seen = set()
    for f in out.glob("*.jsonl"):
        for line in f.open(encoding="utf-8"):
            try: seen.add(json.loads(line)["rptid"])
            except Exception: pass
    s = requests.Session()
    d0, d1 = date.fromisoformat(a.start), date.fromisoformat(a.end)
    d = d1  # 从最近往回抓
    while d >= d0:
        ds = d.isoformat()
        if ds in done: d -= timedelta(days=1); continue
        n_new = 0; p = 1
        while True:
            t = get(s, LIST.format(d=ds, p=p))
            rows = ROW.findall(t) if t else []
            for title, link, rptid, rtype, pub, org, researcher in rows:
                if a.cap and n_new >= a.cap: break
                if rptid in seen: continue
                st = get(s, "https:" + link if link.startswith("//") else link)
                if not st: continue
                text, sym, _ = parse_show(st)
                rec = {"rptid": rptid, "title": html.unescape(title).strip(), "rtype": rtype.strip(), "date": pub,
                       "org": re.sub(r'<[^>]+>', '', org).strip(), "researcher": re.sub(r'<[^>]+>', '', researcher).strip(), "symbol": sym, "text": text}
                with (out / f"{pub[:4]}.jsonl").open("a", encoding="utf-8") as fo:
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                seen.add(rptid); n_new += 1
                time.sleep(a.sleep)
            if len(rows) < 40 or (a.cap and n_new >= a.cap): break
            p += 1; time.sleep(a.sleep)
        with donef.open("a") as fo: fo.write(ds + "\n")
        print(f"{ds} 新增 {n_new} 累计 {len(seen)}", flush=True)
        d -= timedelta(days=1)

if __name__ == "__main__":
    main()
