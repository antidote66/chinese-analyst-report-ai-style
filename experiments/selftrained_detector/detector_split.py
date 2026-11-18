# Re-derive the self-trained detector's rptid-level train/val/test split exactly as
# gen2/train_detector.py did (random.seed(0); shuffle sorted ids; 15% test, 10% val).
import json,random,sys,collections,pandas as pd
G=sys.argv[1]; OUT=sys.argv[2]
random.seed(0)
gen=[json.loads(l) for l in open(G+'data/gen/gen.jsonl',encoding='utf-8')][:1500]
pol=[json.loads(l) for l in open(G+'data/gen/polish.jsonl',encoding='utf-8')][:1500]
hum=json.load(open(G+'data/human_neg.json',encoding='utf-8'))[:len(gen)+len(pol)]
pos=[dict(rptid=d['rptid'],label=1,kind=d['mode']) for d in gen+pol]
neg=[dict(rptid=d['rptid'],label=0,kind='human_2019') for d in hum]
ids=sorted({d['rptid'] for d in pos+neg}); random.shuffle(ids); n=len(ids)
split={i:('test' if k<n*0.15 else 'val' if k<n*0.25 else 'train') for k,i in enumerate(ids)}
rows=[dict(rptid=d['rptid'],label=d['label'],kind=d['kind'],split=split[d['rptid']]) for d in pos+neg]
df=pd.DataFrame(rows); df.to_csv(OUT,index=False)
print(df.groupby(['split','kind']).size().to_dict()); print('test n',(df.split=='test').sum(),'humans in test',((df.split=='test')&(df.label==0)).sum())
print('human_neg years',collections.Counter(d['date'][:4] for d in hum) if 'date' in hum[0] else hum[0].keys())
