import json,glob,numpy as np,pandas as pd,jieba,warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
W=pd.read_csv('/home/claude/sina/out_pilot/candidate_words_labeled_v1.csv',encoding='utf-8-sig')
S=set(w for w,l in zip(W.word,W.label) if str(l).startswith('S')); cyc=set('下滑 下降 亏损 减少 放缓 疲软 低迷 恶化 收缩 回落 转亏'.split())
rows=[]; seen=set()
files=sorted(glob.glob('/home/claude/sina/panel/*.jsonl'))
# Nov 1-4 windows come from the existing base/data files
for y in range(2019,2027):
    for f in [f'/home/claude/sina/base/{y}.jsonl',f'/home/claude/sina/data/{y}.jsonl']: 
        if __import__('os').path.exists(f): files.append(f)
for f in files:
    for l in open(f,encoding='utf-8'):
        try: r=json.loads(l)
        except: continue
        d=r.get('date','')
        if r['rptid'] in seen or len(d)<10: continue
        mo=d[5:7]; day=int(d[8:10])
        if day>4: continue
        if mo=='11' and 'panel' in f: pass
        seen.add(r['rptid']); t=r.get('text','')
        if len(t)<200: continue
        rows.append((d[:7],d,r['rptid'],t))
print('reports in windows',len(rows))
out=[]
for ym,d,rid,t in rows:
    s=set(w for w in jieba.lcut(t) if len(w)>=2)
    out.append((ym,rid,len(s&S)/len(S),len(s&cyc)/len(cyc)))
df=pd.DataFrame(out,columns=['ym','rptid','S','C']); df.to_csv('/home/claude/final/panel_reports.csv',index=False)
g=df.groupby('ym').agg(n=('S','size'),S=('S','mean'),C=('C','mean')).reset_index()
rng=np.random.default_rng(0); lo=[];hi=[]
for ym,sub in df.groupby('ym'):
    v=sub.S.values; b=[rng.choice(v,len(v)).mean() for _ in range(500)]; lo.append(np.percentile(b,2.5)); hi.append(np.percentile(b,97.5))
g['lo']=lo; g['hi']=hi; g.to_csv('/home/claude/final/panel_monthly.csv',index=False); print(g.round(4).to_string())
df['t']=(pd.to_datetime(df.ym+'-01').dt.year-2019)*12+pd.to_datetime(df.ym+'-01').dt.month
df['k23']=np.maximum(df.t-48,0); df['k25']=np.maximum(df.t-72,0)  # 2023-01 -> t=49-... approx
for y in ['S','C']:
    m=smf.ols(f'{y} ~ t + k23 + k25',data=df).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(df.ym)[0]})
    print(y,'slope pre-2023 per month',round(m.params['t'],6),'+',round(m.params['k23'],6),'(2023+) +',round(m.params['k25'],6),'(2025+)')
    for k in ['t','k23','k25']: print('  ',k,round(m.params[k],6),'CI',[round(x,6) for x in m.conf_int().loc[k]])
