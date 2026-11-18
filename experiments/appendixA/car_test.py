import json, sys, pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
y=sys.argv[1]; k=json.load(open(f'k{y}_all.json'))
S={c:pd.Series([float(r.split(',')[1]) for r in v], index=pd.to_datetime([r.split(',')[0] for r in v])) for c,v in k.items() if v}
V={c:pd.Series([float(r.split(',')[2]) for r in v], index=pd.to_datetime([r.split(',')[0] for r in v])) for c,v in k.items() if v}
bm=S['000300']; rep=pd.read_csv(f'rep{y}.csv',dtype={'code':str}); rep['date']=pd.to_datetime(rep.date)
def ret(c,d,h):
    s=S.get(c)
    if s is None: return np.nan
    later=s.index[s.index>=d]
    if len(later)==0: return np.nan
    i=s.index.get_loc(later[0]); j=i+h
    if j<0 or j>=len(s): return np.nan
    bi=bm.index.get_loc(later[0])
    if h>0: return (s.iloc[j]/s.iloc[i]-1)-(bm.iloc[bi+h]/bm.iloc[bi]-1)
    return (s.iloc[i]/s.iloc[j]-1)-(bm.iloc[bi]/bm.iloc[bi+h]-1)
for h in (5,20,60): rep[f'car{h}']=[ret(c,d,h) for c,d in zip(rep.code,rep.date)]
rep['pre20']=[ret(c,d,-20) for c,d in zip(rep.code,rep.date)]
rep['lnamt']=[np.log(V[c].loc[:d].tail(20).mean()) if c in V and len(V[c].loc[:d])>0 else np.nan for c,d in zip(rep.code,rep.date)]
rep['n_rep']=rep.groupby('code').code.transform('size')
for col in ['euph','s_score','pure','neg']: rep['z_'+col]=(rep[col]-rep[col].mean())/rep[col].std()
def run(yv, spec):
    d=rep.dropna(subset=[yv,'pre20','lnamt'])
    m=smf.ols(f'{yv} ~ {spec} + pre20 + lnamt + np.log(n_rep) + C(org) + C(date)', data=d).fit(cov_type='cluster', cov_kwds={'groups':pd.factorize(d.code)[0]})
    return m,len(d)
print(f'===== {y} 窗口  样本 {rep.car20.notna().sum()}  (按股票聚类SE)')
for yv in ['car5','car20','car60']:
    for spec in ['z_s_score + z_neg','z_euph + z_neg','z_pure + z_neg']:
        m,n=run(yv,spec); terms=[t for t in m.params.index if t.startswith('z_')]
        print(f'{yv:6s} n={n}  '+'  '.join(f'{t[2:]}: {m.params[t]*100:+.2f}% (t={m.tvalues[t]:.2f})' for t in terms))
g=rep.groupby(pd.qcut(rep.s_score.rank(method='first'),5,labels=[1,2,3,4,5]))[['car5','car20','car60']].mean()*100
print('s_score 五分位 CAR(%):'); print(g.round(2).to_string()); print('Q5-Q1:', (g.loc[5]-g.loc[1]).round(2).to_dict())
rep.to_csv(f'rep{y}_car.csv',index=False)
