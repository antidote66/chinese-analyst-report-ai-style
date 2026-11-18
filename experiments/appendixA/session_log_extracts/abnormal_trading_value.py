# Extracted verbatim from the session log 2026-11-06T09:28:33.799307Z (inline code run in /home/claude/sina).
# Inputs (price JSON, Eastmoney metadata, corpus) are not in this repository.
import json, pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
out=[]
for y in (2023,2024,2025):
    k=json.load(open(f'k{y}_all.json'))
    A={c:pd.Series([float(r.split(',')[2]) for r in v], index=pd.to_datetime([r.split(',')[0] for r in v])) for c,v in k.items() if v}
    rep=pd.read_csv(f'rep{y}_car.csv',dtype={'code':str}); rep['date']=pd.to_datetime(rep.date)
    rows=[]
    for i,r in rep.iterrows():
        s=A.get(r.code)
        if s is None: rows.append((np.nan,np.nan)); continue
        later=s.index[s.index>=r.date]
        if len(later)==0: rows.append((np.nan,np.nan)); continue
        p=s.index.get_loc(later[0])
        if p-40<0 or p+20>=len(s): rows.append((np.nan,np.nan)); continue
        pre=s.iloc[p-40:p-5].mean(); post=s.iloc[p+4:p+21].mean()
        rows.append((np.log(post/pre) if pre>0 and post>0 else np.nan, 0))
    rep['atv']=[x[0] for x in rows]
    # non-overlap: no other report for same stock within next 20 calendar days (28)
    rep=rep.sort_values(['code','date']); nxt=rep.groupby('code').date.shift(-1)
    rep['nonover']=(nxt.isna())|((nxt-rep.date).dt.days>28)
    rep['year']=y; out.append(rep)
d=pd.concat(out)
print('all',d.atv.notna().sum(),'nonoverlap',(d.atv.notna()&d.nonover).sum())
for lab,dd in [('全样本',d),('无重叠窗口',d[d.nonover])]:
    for y in [None,2023,2024,2025]:
        x=(dd if y is None else dd[dd.year==y]).dropna(subset=['atv','pre20','lnamt'])
        if len(x)<200: continue
        m=smf.ols('atv ~ z_s_score + z_neg + pre20 + lnamt + np.log(n_rep) + C(org) + C(date)',data=x).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(x.code)[0]})
        b=m.params['z_s_score']; se=m.bse['z_s_score']
        print(f'{lab} {y or "合并"} n={len(x)} stocks={x.code.nunique()} 风格词密度+1SD -> 异常成交额 {b:+.3f} (SE {se:.3f}, t={m.tvalues["z_s_score"]:.2f}, 95%CI [{b-1.96*se:+.3f},{b+1.96*se:+.3f}])')
