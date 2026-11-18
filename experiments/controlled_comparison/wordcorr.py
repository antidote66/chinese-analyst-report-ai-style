import json,glob,numpy as np,pandas as pd,jieba
from scipy.stats import spearmanr
W=pd.read_csv('/home/claude/sina/out_pilot/candidate_words_labeled_v1.csv',encoding='utf-8-sig')
S=W[W.label.astype(str).str.startswith('S')].reset_index(drop=True)
sk={d['rptid']:d for d in json.load(open('/home/claude/gen/data/gen_skeletons_2019.json',encoding='utf-8'))}
def df(texts):
    c=np.zeros(len(S))
    for t in texts:
        s=set(w for w in jieba.lcut(t) if len(w)>=2); c+=[w in s for w in S.word]
    return c/len(texts)
res=[]
for name in ['gen','polish','rewrite','gen_qwen','polish_qwen','rewrite_qwen']:
    rows=[json.loads(l) for l in open(f'/home/claude/gen/data/gen/{name}.jsonl',encoding='utf-8')]
    rows=[r for r in rows if r['rptid'] in sk and len(r['text'])>80]
    g=df([r['text'] for r in rows]); h=df([sk[r['rptid']]['text'] for r in rows])
    d=g-h; rho,p=spearmanr(d,S.delta)
    top=S.assign(d=d).sort_values('d',ascending=False).head(10)
    cover=((g>=0.02)&(S.delta>0.05)).sum()/ (S.delta>0.05).sum()
    res.append((name,round(rho,3),round(p,4),round(cover,3)))
    print(name,'spearman(模型偏好增量, 2026相对2019增量)=',round(rho,3),'p=',round(p,4),'; 2026增长>5pp的词中被模型以>=2%频率使用的比例',round(cover,3))
pd.DataFrame(res,columns=['cond','rho','p','coverage']).to_csv('wordcorr.csv',index=False)
