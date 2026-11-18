#!/usr/bin/env python3
"""
新浪财经研报库 摘要抓取（按发布日期）。
用法: python sina_abstracts.py --start 2023-10-01 --end 2023-11-04 --out data --sleep 0.3 [--max-per-day 0]
输出: {out}/{year}.jsonl  每行 {rptid,title,rtype,date,org,analyst,text}
断点续传: {out}/done_dates.txt 记录已完成日期；重跑同一命令自动跳过。
"""
import argparse, datetime as dt, json, os, re, sys, time
import requests

LIST = "https://stock.finance.sina.com.cn/stock/go.php/vReport_List/kind/search/index.phtml"
SHOW = "https://stock.finance.sina.com.cn/stock/go.php/vReport_Show/kind/search/rptid/{}/index.phtml"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
ROW = re.compile(
    r'rptid/(\d+)/index\.phtml">\s*(.*?)\s*</a>\s*</td>\s*<td>(.*?)</td>\s*<td>(\d{4}-\d{2}-\d{2})</td>'
    r'.*?<span>(.*?)</span>.*?<td><div class="fname"><span>(.*?)</span>', re.S)


def get(s, url, params=None, tries=4):
    for i in range(tries):
        try:
            r = s.get(url, params=params, headers=UA, timeout=25)
            if r.status_code == 200:
                r.encoding = "gbk"
                return r.text
        except requests.RequestException:
            pass
        time.sleep(2 * (i + 1))
    return ""


def list_day(s, day, sleep):
    rows, p = [], 1
    while True:
        h = get(s, LIST, {"t1": "6", "pubdate": day, "p": p})
        found = ROW.findall(h)
        if not found:
            break
        for rid, title, rtype, date, org, analyst in found:
            if date == day:
                rows.append(dict(rptid=rid, title=title.strip(), rtype=rtype.strip(), date=date,
                                 org=org.strip(), analyst=analyst.strip()))
        if len(found) < 40:
            break
        p += 1
        time.sleep(sleep)
    seen, out = set(), []
    for r in rows:
        if r["rptid"] not in seen:
            seen.add(r["rptid"]); out.append(r)
    return out


def abstract(s, rid):
    h = get(s, SHOW.format(rid))
    m = re.search(r'<div class="blk_container">(.*?)</div>', h, re.S)
    if not m:
        return ""
    t = re.sub(r"<[^>]+>", "", m.group(1))
    t = re.sub(r"&nbsp;|　", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--out", default="data"); ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--max-per-day", type=int, default=0, help="每天最多抓多少篇摘要，0=全部")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    done_f = os.path.join(a.out, "done_dates.txt")
    done = set(open(done_f, encoding="utf-8").read().split()) if os.path.exists(done_f) else set()
    d0, d1 = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    s = requests.Session(); total = 0
    day = d1
    while day >= d0:                      # 从新到旧
        ds = day.isoformat()
        if ds in done:
            day -= dt.timedelta(days=1); continue
        rows = list_day(s, ds, a.sleep)
        if a.max_per_day:
            rows = rows[: a.max_per_day]
        n = 0
        with open(os.path.join(a.out, f"{ds[:4]}.jsonl"), "a", encoding="utf-8") as f:
            for r in rows:
                r["text"] = abstract(s, r["rptid"])
                if len(r["text"]) >= 80:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
                time.sleep(a.sleep)
        total += n
        with open(done_f, "a", encoding="utf-8") as f:
            f.write(ds + "\n")
        print(f"{ds} 列表{len(rows)} 摘要{n} 累计{total}", flush=True)
        day -= dt.timedelta(days=1)


if __name__ == "__main__":
    main()
