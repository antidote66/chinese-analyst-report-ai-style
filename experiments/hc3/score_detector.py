import json, re, sys, torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
torch.set_num_threads(2)
name="Hello-SimpleAI/chatgpt-detector-roberta-chinese"
tok=AutoTokenizer.from_pretrained(name); mdl=AutoModelForSequenceClassification.from_pretrained(name).eval()
docs=[]
for src in ['mac/2026.jsonl','data/2026.jsonl']+[f'base/{y}.jsonl' for y in range(2019,2026)]:
    for line in open(src,encoding='utf-8'):
        d=json.loads(line)
        if len(d.get('text',''))<80 or not ('10-01'<=d['date'][5:]<='11-04'): continue
        docs.append((d['rptid'],d['date'][:4],d['rtype'],d['org'],d['text']))
seen=set(); docs=[d for d in docs if not (d[0] in seen or seen.add(d[0]))]
import random; random.seed(7)
by={}
for d in docs: by.setdefault(d[1],[]).append(d)
docs=[d for y in sorted(by) for d in random.sample(by[y],min(1100,len(by[y])))]
print('docs',len(docs),flush=True)
out=open('out_pilot/detector_scores_strat.csv','w',encoding='utf-8'); out.write('rptid,year,rtype,org,p_ai\n')
B=16
with torch.no_grad():
    for i in range(0,len(docs),B):
        batch=docs[i:i+B]
        enc=tok([d[4] for d in batch],return_tensors='pt',padding=True,truncation=True,max_length=512)
        p=torch.softmax(mdl(**enc).logits,-1)[:,1].tolist()
        for d,pp in zip(batch,p): out.write(f'{d[0]},{d[1]},{d[2]},"{d[3]}",{pp:.5f}\n')
        if i%1600==0: out.flush(); print(i,flush=True)
out.close(); print('done',flush=True)
