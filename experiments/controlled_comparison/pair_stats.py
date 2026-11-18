# Per-pair word-occurrence stats for the controlled comparison (no text), and a check
# that they reproduce final/cross_model_results.csv (paper §5.5) exactly.
import json,sys,numpy as np,pandas as pd,jieba
jieba.setLogLevel(60)
G=sys.argv[1]; LAB=sys.argv[2]; OUT=sys.argv[3]; REF=sys.argv[4]
W=pd.read_csv(LAB,encoding='utf-8-sig'); S=[w for w,l in zip(W.word,W.label) if str(l).startswith('S')]
cyc=['下滑','下降','亏损','减少','放缓','疲软','低迷','恶化','收缩','回落','转亏']
sk={d['rptid']:d for d in json.load(open(G+'gen_skeletons_2019.json',encoding='utf-8'))}
def toks(t): return set(w for w in jieba.lcut(t) if len(w)>=2)
rows=[]; res=[]
for name,model,cond in [('gen','DeepSeek','独立生成'),('gen_qwen','千问','独立生成'),('polish','DeepSeek','轻度润色'),('polish_qwen','千问','轻度润色'),('rewrite','DeepSeek','深度改写'),('rewrite_qwen','千问','深度改写')]:
    R=[json.loads(l) for l in open(G+f'gen/{name}.jsonl',encoding='utf-8')]
    R=[r for r in R if r['rptid'] in sk and len(r['text'])>80]
    gs=[];hs=[]
    for r in R:
        a=toks(r['text']); b=toks(sk[r['rptid']]['text'])
        rec=dict(condition_file=name,model=model,intervention=cond,rptid=r['rptid'],model_id=r.get('model',''),
                 model_chars=len(r['text']),human_chars=len(sk[r['rptid']]['text']),
                 S_model=sum(w in a for w in S)/len(S),S_human=sum(w in b for w in S)/len(S),
                 cyc_model=sum(w in a for w in cyc)/len(cyc),cyc_human=sum(w in b for w in cyc)/len(cyc))
        rows.append(rec)
    d=pd.DataFrame([x for x in rows if x['condition_file']==name])
    np.random.seed(0)  # note: original script seeded once and looped conditions in glob order
    res.append((name,len(d),d.S_model.mean()/d.S_human.mean()))
P=pd.DataFrame(rows); P.to_csv(OUT,index=False)
ref=pd.read_csv(REF).set_index('cond')
for n,k,r in res: print(n,k,round(r,6),'ref',ref.loc[n,'n'],round(ref.loc[n,'ratio'],6),'OK' if k==ref.loc[n,'n'] and abs(r-ref.loc[n,'ratio'])<1e-9 else 'DIFF')
