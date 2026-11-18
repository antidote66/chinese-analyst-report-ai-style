import json,glob,os,numpy as np,pandas as pd,statsmodels.formula.api as smf,warnings; warnings.filterwarnings('ignore')
P=pd.read_csv('/home/claude/final/panel_reports.csv',dtype={'rptid':str}).rename(columns={'C':'Cyc'})
meta={}; order={}
files=sorted(glob.glob('/home/claude/sina/panel/*.jsonl'))+[f for y in range(2019,2027) for f in (f'/home/claude/sina/base/{y}.jsonl',f'/home/claude/sina/data/{y}.jsonl') if os.path.exists(f)]
for f in files:
    cnt={}
    for l in open(f,encoding='utf-8'):
        try: r=json.loads(l)
        except: continue
        d=r.get('date','')
        if r['rptid'] in meta: continue
        cnt[d]=cnt.get(d,0)+1
        meta[r['rptid']]=(r.get('org',''),r.get('rtype',''),d); order[r['rptid']]=cnt[d]
P['org']=P.rptid.map(lambda i:meta.get(i,('','',''))[0]); P['rtype']=P.rptid.map(lambda i:meta.get(i,('','',''))[1]); P['date']=P.rptid.map(lambda i:meta.get(i,('','',''))[2]); P['rank']=P.rptid.map(order)
top=P.org.value_counts().index[:60]; P['orgc']=np.where(P.org.isin(top),P.org,'其他')
P['rt']=P.rtype.replace('',"未知")
print('rtype counts',P.rt.value_counts().head(8).to_dict())
P['t']=(P.ym.str[:4].astype(int)-2019)*12+P.ym.str[5:7].astype(int)
P['k23']=np.maximum(P.t-48,0); P['k25']=np.maximum(P.t-72,0)
# (1) list-order check in Nov full windows
nov=P[P.ym.str[5:7]=='11']
r=nov.groupby('ym').apply(lambda g: pd.Series({'全量n':len(g),'全量':g.S.mean(),'前100n':(g['rank']<=100).sum(),'前100':g[g['rank']<=100].S.mean()}))
r['差']=r['前100']-r['全量']; print(r.round(4).to_string()); print('mean abs diff',r['差'].abs().mean().round(4))
# (2) FE regressions, report level, cluster by month
for spec,lab in [('S ~ t + k23 + k25','无固定效应'),('S ~ t + k23 + k25 + C(orgc)','券商FE'),('S ~ t + k23 + k25 + C(orgc) + C(rt)','券商+报告类型FE')]:
    m=smf.ols(spec,data=P).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(P.ym)[0]})
    ci=m.conf_int()
    print(lab, {k:f'{m.params[k]:.5f} [{ci.loc[k,0]:.5f},{ci.loc[k,1]:.5f}]' for k in ['t','k23','k25']})
# within report type: company reports only
for rt in P.rt.value_counts().index[:3]:
    q=P[P.rt==rt]; m=smf.ols('S ~ t + k23 + k25 + C(orgc)',data=q).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(q.ym)[0]})
    print('仅',rt,len(q),f"k25 {m.params['k25']:.5f} [{m.conf_int().loc['k25',0]:.5f},{m.conf_int().loc['k25',1]:.5f}]")
# (3) placebo: monthly equal-weight series, single-knot slope change, pre-2023 fake knots vs 2025-01 knot in 2023-2026
g=pd.read_csv('/home/claude/final/panel_monthly.csv'); g['t']=range(1,len(g)+1); g=g[(g.n>=150)&(g.ym.str[5:7]!='11')]
def knot(df,b):
    df=df.copy(); df['kb']=np.maximum(df.t-b,0); m=smf.ols('S ~ t + kb',data=df).fit(cov_type='HAC',cov_kwds={'maxlags':3}); return m.params['kb'],m.tvalues['kb']
pre=g[g.t<=48]; post=g[g.t>48]
pl=[knot(pre,b) for b in range(7,43)]
real=knot(post,72)
print('placebo knots 2019-07..2022-06 (pre-2023 data): slope change range',round(min(x[0] for x in pl),5),round(max(x[0] for x in pl),5),'max |t|',round(max(abs(x[1]) for x in pl),2))
print('pre-specified 2025-01 knot on 2023-2026 data: change',round(real[0],5),'t',round(real[1],2))
