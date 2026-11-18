import json,re,os,sys,numpy as np,pandas as pd,jieba
from scipy.stats import spearmanr
rng=np.random.default_rng(0)
W=pd.read_csv('/home/claude/sina/out_pilot/candidate_words_labeled_v1.csv',encoding='utf-8-sig')
SW=W[W.label.astype(str).str.startswith('S')].reset_index(drop=True); S=set(SW.word)
sk={d['rptid']:d for d in json.load(open('/home/claude/gen/data/gen_skeletons_2019.json',encoding='utf-8'))}
G='/home/claude/gen/data/gen/'
M={'DeepSeek':{'润色':'polish','深度改写':'rewrite','独立生成':'gen','中性改写':'rewriteN','生成重复2':'gen_r2','生成重复3':'gen_r3'},
   '千问':{'润色':'polish_qwen','深度改写':'rewrite_qwen','独立生成':'gen_qwen','中性改写':'rewriteN_qwen','生成重复2':'gen_qwen_r2','生成重复3':'gen_qwen_r3'},
   'Claude':{'润色':'polish_claude','深度改写':'rewrite_claude','独立生成':'gen_claude','中性改写':'rewriteN_claude','生成重复2':'gen_claude_r2'}}
def load(f):
    D={}
    if not os.path.exists(G+f+'.jsonl'): return D
    for l in open(G+f+'.jsonl',encoding='utf-8'):
        r=json.loads(l)
        trunc=(r.get('usage',{}).get('completion_tokens',0)>=895 and not r.get('regen_cap3000') and 'claude' not in f) or r.get('stop')=='max_tokens'
        if r['rptid'] in sk and len(r['text'])>80 and not trunc: D[r['rptid']]=r['text']
    return D
D={(m,c):load(f) for m in M for c,f in M[m].items()}
cache={}
def toks(t):
    if t not in cache: cache[t]=[w for w in jieba.lcut(t) if len(w)>=2]
    return cache[t]
def docS(t): return len(set(toks(t))&S)/len(S)
def per1k(t): return 1000*sum(w in S for w in toks(t))/max(len(t),1)
num=re.compile(r'\d+(?:\.\d+)?')
def nums(t): return set(x for x in num.findall(t) if len(x)>=2 or '.' in x)
def boot(a,b):
    n=len(a); r=[a[x].mean()/b[x].mean() for x in (rng.integers(0,n,n) for _ in range(2000))]
    return a.mean()/b.mean(),np.percentile(r,2.5),np.percentile(r,97.5)
def fmt(t): return f'{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]'
def stats(m,c,ids):
    T=D[(m,c)]; H={i:sk[i]['text'] for i in ids}
    gd=np.array([docS(T[i]) for i in ids]); hd=np.array([docS(H[i]) for i in ids])
    gk=np.array([per1k(T[i]) for i in ids]); hk=np.array([per1k(H[i]) for i in ids])
    td=[];th=[]
    for i in ids:
        L=min(len(T[i]),len(H[i])); td.append(docS(T[i][:L])); th.append(docS(H[i][:L]))
    gl=np.mean([len(T[i]) for i in ids])/np.mean([len(H[i]) for i in ids])
    gen='生成' in c
    src={i:(nums(' '.join(sk[i]['facts'])) if gen else nums(H[i])) for i in ids}
    keep=np.mean([len(nums(T[i])&src[i])/len(src[i]) for i in ids if src[i]])
    new=np.mean([len(nums(T[i])-nums(H[i])-nums(' '.join(sk[i]['facts'])))/max(len(nums(T[i])),1) for i in ids])
    return dict(模型=m,条件=c,n=len(ids),长度比=round(gl,2),文档出现率比=fmt(boot(gd,hd)),每千字频次比=fmt(boot(gk,hk)),截断等长比=fmt(boot(np.array(td),np.array(th))),数字保留率=round(keep,3),新增数字占比=round(new,3),_doc=boot(gd,hd),_k=boot(gk,hk),_t=boot(np.array(td),np.array(th)))
main=['润色','深度改写','独立生成']
avail=[(m,c) for m in M for c in main if len(D.get((m,c),{}))>0]
common3=sorted(set.intersection(*[set(D[k]) for k in avail]))
print('main conditions',avail,'common ids',len(common3))
rows=[stats(m,c,common3) for m,c in avail]
T1=pd.DataFrame(rows); T1.drop(columns=['_doc','_k','_t']).to_csv('/home/claude/final/table_main.csv',index=False); print(T1.drop(columns=['_doc','_k','_t']).to_string())
T1.to_pickle('/home/claude/final/table_main.pkl')
# prompt sensitivity: rewrite vs rewriteN on common ids per model
out=[]
for m in M:
    ids=sorted(set(D[(m,'深度改写')])&set(D[(m,'中性改写')]))
    if len(ids)<100: continue
    for c in ['深度改写','中性改写']: r=stats(m,c,ids); out.append({k:v for k,v in r.items() if not k.startswith('_')})
T2=pd.DataFrame(out); print(T2.to_string()); T2.to_csv('/home/claude/final/table_prompt.csv',index=False)
# repeats
out=[]
for m in M:
    reps=[c for c in ['独立生成','生成重复2','生成重复3'] if len(D.get((m,c),{}))>0]
    ids=sorted(set.intersection(*[set(D[(m,c)]) for c in reps]))
    if len(ids)<100: continue
    vals=[stats(m,c,ids) for c in reps]
    out.append(dict(模型=m,n=len(ids),次数=len(reps),文档出现率比=' / '.join(f"{v['_doc'][0]:.3f}" for v in vals),每千字频次比=' / '.join(f"{v['_k'][0]:.3f}" for v in vals)))
T3=pd.DataFrame(out); print(T3.to_string()); T3.to_csv('/home/claude/final/table_repeat.csv',index=False)
# word-level spearman vs real 2019->2026 delta (on common3)
out=[]
for m,c in avail:
    T=D[(m,c)]
    g=np.array([[w in set(toks(T[i])) for w in SW.word] for i in common3]).mean(0)
    h=np.array([[w in set(toks(sk[i]['text'])) for w in SW.word] for i in common3]).mean(0)
    rho,p=spearmanr(g-h,SW.delta); out.append(dict(模型=m,条件=c,rho=round(rho,3),p=round(p,4)))
T4=pd.DataFrame(out); print(T4.to_string()); T4.to_csv('/home/claude/final/table_wordcorr.csv',index=False)
words=['承压','修复','扰动','兑现','韧性','闭环','适配','曲线','赋能']
out=[]
H=[set(toks(sk[i]['text'])) for i in common3]
out.append(dict(来源='2019 人类原文',**{w:round(100*np.mean([w in s for s in H]),1) for w in words}))
for m,c in avail:
    Ts=[set(toks(D[(m,c)][i])) for i in common3]
    out.append(dict(来源=f'{m} {c}',**{w:round(100*np.mean([w in s for s in Ts]),1) for w in words}))
T5=pd.DataFrame(out); print(T5.to_string()); T5.to_csv('/home/claude/final/table_words.csv',index=False)
