#!/usr/bin/env python3
"""
中文研报 AI 文本检测器：hfl/chinese-roberta-wwm-ext 微调。
正样本：DeepSeek 生成(gen) + 润色(polish)；负样本：2019–2022 人类研报。
按 rptid 划分 train/val/test（同一骨架的 gen/polish/原文不跨集）。
输出：模型 ckpt/、评估 out/detector_eval.json、2026 年研报打分 out/scores_2026.csv、
      以及按年分层的人类研报打分 out/scores_by_year.csv（2019–2025 各 400 篇）。
"""
import json, random, sys, os, time, torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, f1_score
torch.set_num_threads(2); random.seed(0); torch.manual_seed(0)
NAME = "hfl/chinese-roberta-wwm-ext"; MAXLEN = 320; BS = 8; EPOCHS = 2; LR = 2e-5
NPOS = int(sys.argv[1]) if len(sys.argv) > 1 else 1500   # 每类正样本上限
os.makedirs('out', exist_ok=True)

gen = [json.loads(l) for l in open('data/gen/gen.jsonl', encoding='utf-8')][:NPOS]
pol = [json.loads(l) for l in open('data/gen/polish.jsonl', encoding='utf-8')][:NPOS] if os.path.exists('data/gen/polish.jsonl') else []
hum = json.load(open('data/human_neg.json', encoding='utf-8'))[:len(gen) + len(pol)]
pos = [dict(rptid=d['rptid'], text=d['text'], label=1, kind=d['mode']) for d in gen + pol]
neg = [dict(rptid=d['rptid'], text=d['text'], label=0, kind='human') for d in hum]
ids = sorted({d['rptid'] for d in pos + neg}); random.shuffle(ids)
n = len(ids); split = {i: ('test' if k < n * 0.15 else 'val' if k < n * 0.25 else 'train') for k, i in enumerate(ids)}
data = {s: [d for d in pos + neg if split[d['rptid']] == s] for s in ('train', 'val', 'test')}
print({s: (len(v), sum(d['label'] for d in v)) for s, v in data.items()}, flush=True)

tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=2)
for p in model.bert.embeddings.parameters(): p.requires_grad=False
for layer in model.bert.encoder.layer[:6]:
    for p in layer.parameters(): p.requires_grad=False
def collate(batch):
    enc = tok([b['text'] for b in batch], padding=True, truncation=True, max_length=MAXLEN, return_tensors='pt')
    enc['labels'] = torch.tensor([b['label'] for b in batch]); return enc
def predict(items):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(items), 16):
            enc = tok([b['text'] for b in items[i:i+16]], padding=True, truncation=True, max_length=MAXLEN, return_tensors='pt')
            out += torch.softmax(model(**enc).logits, -1)[:, 1].tolist()
    return out
def evaluate(items, tag):
    p = predict(items); y = [d['label'] for d in items]
    res = dict(n=len(items), auc=roc_auc_score(y, p), f1=f1_score(y, [int(x > 0.5) for x in p]))
    for k in sorted({d['kind'] for d in items}):
        pk = [x for x, d in zip(p, items) if d['kind'] == k]; res[f'mean_p_{k}'] = float(np.mean(pk)); res[f'rate_{k}'] = float(np.mean([x > 0.5 for x in pk]))
    print(tag, {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}, flush=True); return res, p

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
dl = DataLoader(data['train'], batch_size=BS, shuffle=True, collate_fn=collate)
sch = get_linear_schedule_with_warmup(opt, int(0.03 * len(dl) * EPOCHS), len(dl) * EPOCHS // 2)
best = 0; t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    for i, b in enumerate(dl):
        loss = model(**b).loss / 2; loss.backward()
        if i % 2 == 1: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad()
        if i % 20 == 0: print(f'ep{ep} step{i}/{len(dl)} loss {loss.item():.4f} {int(time.time()-t0)}s', flush=True)
    res, _ = evaluate(data['val'], f'val ep{ep}')
    if res['auc'] > best: best = res['auc']; model.save_pretrained('ckpt'); tok.save_pretrained('ckpt')
model = AutoModelForSequenceClassification.from_pretrained('ckpt')
test_res, _ = evaluate(data['test'], 'TEST')
json.dump(test_res, open('out/detector_eval.json', 'w'), indent=1)

# 应用：2026 年研报
T = json.load(open('data/test_2026.json', encoding='utf-8')); p = predict(T)
pd.DataFrame(dict(rptid=[d['rptid'] for d in T], org=[d['org'] for d in T], p_ai=p)).to_csv('out/scores_2026.csv', index=False)
print('2026: mean p', round(float(np.mean(p)), 4), 'rate>0.5', round(float(np.mean([x > 0.5 for x in p])), 4), flush=True)
# 按年分层（人类研报池里不在训练集的）
rows = []
for y in range(2020, 2026):
    pool = [json.loads(l) for l in open(f'/home/claude/sina/base/{y}.jsonl', encoding='utf-8')]
    pool = [d for d in pool if d['rtype'] == '公司' and 400 <= len(d['text']) <= 2000 and split.get(d['rptid'], 'test') != 'train'][:400]
    pp = predict(pool); rows.append(dict(year=y, n=len(pool), mean_p=float(np.mean(pp)), rate=float(np.mean([x > 0.5 for x in pp]))))
rows.append(dict(year=2026, n=len(p), mean_p=float(np.mean(p)), rate=float(np.mean([x > 0.5 for x in p]))))
pd.DataFrame(rows).to_csv('out/scores_by_year.csv', index=False); print(pd.DataFrame(rows).round(4).to_string(index=False), flush=True)
