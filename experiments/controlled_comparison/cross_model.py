import json, glob, re, numpy as np, pandas as pd, jieba, sys, random
random.seed(0); np.random.seed(0)
W=pd.read_csv('/home/claude/sina/out_pilot/candidate_words_labeled_v1.csv',encoding='utf-8-sig')
S=[w for w,l in zip(W.word,W.label) if str(l).startswith('S')]
E_=[w for w,l in zip(W.word,W.label) if str(l).startswith('E')]
cyc=['下滑','下降','亏损','减少','放缓','疲软','低迷','恶化','收缩','回落','转亏']
sk={d['rptid']:d for d in json.load(open('/home/claude/gen/data/gen_skeletons_2019.json',encoding='utf-8'))}
def toks(t): return set(w for w in jieba.lcut(t) if len(w)>=2)
def vec(t,words):
    s=toks(t); return np.array([w in s for w in words],float)
conds={}
for f in sorted(glob.glob('/home/claude/gen/data/gen/*.jsonl')):
    name=f.split('/')[-1][:-6]
    if 'test' in name or name.endswith('_pro'): continue
    rows=[json.loads(l) for l in open(f,encoding='utf-8')]
    rows=[r for r in rows if r['rptid'] in sk and len(r['text'])>80]
    if len(rows)>=100: conds[name]=rows
out=[]
for name,rows in conds.items():
    gs=np.array([vec(r['text'],S).mean() for r in rows]); hs=np.array([vec(sk[r['rptid']]['text'],S).mean() for r in rows])
    gc=np.array([vec(r['text'],cyc).mean() for r in rows]); hc=np.array([vec(sk[r['rptid']]['text'],cyc).mean() for r in rows])
    n=len(rows); idx=np.arange(n); rat=[]
    for _ in range(2000):
        b=np.random.choice(idx,n); rat.append(gs[b].mean()/hs[b].mean())
    GV=np.mean([vec(r['text'],S) for r in rows],0); HV=np.mean([vec(sk[r['rptid']]['text'],S) for r in rows],0)
    out.append(dict(cond=name,n=n,S_gen=gs.mean(),S_human=hs.mean(),ratio=gs.mean()/hs.mean(),lo=np.percentile(rat,2.5),hi=np.percentile(rat,97.5),cyc_ratio=gc.mean()/hc.mean()))
    top=sorted([(GV[i]/max(HV[i],1/n),S[i],GV[i],HV[i]) for i in range(len(S)) if GV[i]>=0.05],reverse=True)[:8]
    print(name,'top words',[(w,round(r,1)) for r,w,_,_ in top])
df=pd.DataFrame(out); print(df.round(3).to_string()); df.to_csv('/home/claude/final/cross_model_results.csv',index=False)
