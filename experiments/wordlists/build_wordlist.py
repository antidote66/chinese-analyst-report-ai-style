import json,re,collections,sys
import pandas as pd,numpy as np,jieba
from scipy.stats import spearmanr
CJK=re.compile(r"^[一-鿿]{2,}$")
SRC={y:f'base/{y}.jsonl' for y in range(2019,2026)}; SRC[2026]=['mac/2026.jsonl','data/2026.jsonl']
docs={};seen=set()
for y,p in SRC.items():
    for f in ([p] if isinstance(p,str) else p):
        for l in open(f,encoding='utf-8'):
            try:d=json.loads(l)
            except: continue
            if d['rptid'] in seen or len(d.get('text',''))<80: continue
            if not('10-01'<=d['date'][5:]<='11-04') or d['date'][:4]!=str(y): continue
            seen.add(d['rptid']);docs.setdefault(y,[]).append(d['text'])
N={y:len(v) for y,v in docs.items()};print(N,flush=True)
cnt={y:collections.Counter() for y in docs}
for y,v in docs.items():
    for t in v: cnt[y].update({w for w in jieba.cut(t) if CJK.match(w)})
yrs=sorted(docs)
words={w for y in yrs for w,c in cnt[y].items() if c/N[y]>=0.01}
rows=[]
for w in words:
    f=[cnt[y][w]/N[y] for y in yrs]
    b=np.mean(f[:4]);late=np.mean(f[6:]);early=np.mean(f[:6])
    rows.append(dict(word=w,**{str(y):round(v,4) for y,v in zip(yrs,f)},base1922=round(b,4),late=round(late,4),delta=round(late-early,4),ratio=round(late/max(early,0.003),2),rho=round(spearmanr(yrs,f)[0],2)))
df=pd.DataFrame(rows);df.to_csv('wordlist/all_words_yearly.csv',index=False,encoding='utf-8-sig')
print(len(df))
