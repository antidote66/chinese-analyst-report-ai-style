#!/usr/bin/env python3
"""
零样本 AI 文本检测（Binoculars, Hans et al. 2024）在研报摘要上的打分。
  logppl     = observer 模型的 log 困惑度（越低越"像模型写的"）
  xppl       = observer 与 performer 的交叉困惑度
  binoculars = logppl / xppl（Binoculars 分数，越低越可能是 AI 生成）
  mean_rank / top10 = GLTR 风格的 token 排名统计
用法: python zeroshot_detect.py --data data --years 2019,2023,2026 --per-year 300 --out out
断点续传：out/zeroshot_scores.csv 已有的 rptid 跳过；各年交替打分，随时可比。
"""
import argparse, csv, json, os, random, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

OBS = os.environ.get("OBS_MODEL", "Qwen/Qwen2.5-1.5B")
PERF = os.environ.get("PERF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
MAXTOK = int(os.environ.get("MAXTOK", "512"))


def load(name):
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    m.eval()
    return tok, m


@torch.no_grad()
def score(text, tok, mo, mp):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=MAXTOK).input_ids
    if ids.shape[1] < 16:
        return None
    lo = mo(ids).logits[0, :-1]
    lp = mp(ids).logits[0, :-1]
    tgt = ids[0, 1:]
    nll_sum = xent_sum = 0.0; rank_sum = 0.0; top10 = 0
    for s in range(0, lo.shape[0], 64):                    # 分块，省内存
        o = lo[s:s + 64].float(); p = lp[s:s + 64].float(); t = tgt[s:s + 64]
        lso = torch.log_softmax(o, -1)
        nll_sum += -lso.gather(1, t[:, None]).sum().item()  # observer NLL
        pp = torch.softmax(p, -1)                          # performer 分布
        xent_sum += -(pp * lso).sum().item()               # 交叉熵
        ranks = (o > o.gather(1, t[:, None])).sum(1) + 1
        rank_sum += ranks.float().sum().item(); top10 += (ranks <= 10).sum().item()
        del o, p, lso, pp
    n = lo.shape[0]
    logppl, xent = nll_sum / n, xent_sum / n
    return dict(logppl=round(logppl, 4), xppl=round(xent, 4), binoculars=round(logppl / xent, 4),
                mean_rank=round(rank_sum / n, 2), top10=round(top10 / n, 4), ntok=int(ids.shape[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--years", default="2019,2021,2023,2024,2025,2026")
    ap.add_argument("--per-year", type=int, default=300); ap.add_argument("--out", default="out")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    years = [y.strip() for y in a.years.split(",")]
    rnd = random.Random(a.seed)
    by = {}
    for y in years:
        p = os.path.join(a.data, f"{y}.jsonl")
        if not os.path.exists(p):
            print("缺数据", p, flush=True); continue
        docs = [json.loads(l) for l in open(p, encoding="utf-8")]
        docs = [d for d in docs if len(d.get("text", "")) >= 200]
        rnd.shuffle(docs); by[y] = docs[: a.per_year]
        print(f"{y}: 可用 {len(docs)} 取 {len(by[y])}", flush=True)
    out_f = os.path.join(a.out, "zeroshot_scores.csv")
    done = set()
    if os.path.exists(out_f):
        done = {r["rptid"] for r in csv.DictReader(open(out_f, encoding="utf-8"))}
    order = [(y, by[y][i]) for i in range(max((len(v) for v in by.values()), default=0)) for y in by if i < len(by[y])]
    todo = [(y, d) for y, d in order if d["rptid"] not in done]
    print(f"待打分 {len(todo)} 已完成 {len(done)}  observer={OBS} performer={PERF}", flush=True)
    if not todo:
        return
    print("加载模型（首次需下载约 6 GB）...", flush=True)
    tok, mo = load(OBS); _, mp = load(PERF)
    fields = ["rptid", "year", "date", "org", "rtype", "ntok", "logppl", "xppl", "binoculars", "mean_rank", "top10"]
    new = not os.path.exists(out_f) or os.path.getsize(out_f) == 0
    f = open(out_f, "a", newline="", encoding="utf-8"); w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()
    t0 = time.time()
    for i, (y, d) in enumerate(todo, 1):
        r = score(d["text"], tok, mo, mp)
        if r:
            w.writerow(dict(rptid=d["rptid"], year=y, date=d["date"], org=d["org"], rtype=d["rtype"], **r)); f.flush()
        if i % 10 == 0:
            el = time.time() - t0
            print(f"{i}/{len(todo)}  {el/i:.1f}s/篇  预计剩余 {(len(todo)-i)*el/i/60:.0f} 分钟", flush=True)
    f.close(); print("完成", flush=True)


if __name__ == "__main__":
    main()
