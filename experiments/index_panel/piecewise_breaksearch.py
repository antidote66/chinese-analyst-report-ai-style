# Extracted verbatim from the 2026-11-06 session log (inline command, run in /home/claude/final).
# Equal-weighted 79-month piecewise regression (Newey-West, maxlags=3) and second-break grid search (paper §4.3/§5.2).
import pandas as pd,numpy as np,statsmodels.formula.api as smf
g=pd.read_csv('panel_monthly.csv'); g['t']=range(1,len(g)+1)
g['mo']=g.ym.str[5:7]
h=g[(g.n>=150)&(g.mo!='11')].copy()   # equal-weight months, drop Nov full windows and tiny holiday windows
h['k23']=np.maximum(h.t-48,0); h['k25']=np.maximum(h.t-72,0)
m=smf.ols('S ~ t + k23 + k25',data=h).fit(cov_type='HAC',cov_kwds={'maxlags':3})
print('equal-weight months n=',len(h)); print(m.params.round(6).to_dict()); print(m.conf_int().round(6))
# break-date grid: S ~ t + k23 + max(t-b,0)
res=[]
for b in range(55,90):
    h['kb']=np.maximum(h.t-b,0); mm=smf.ols('S ~ t + k23 + kb',data=h).fit(); res.append((b,g.ym[b-1],mm.ssr,mm.params['kb']))
r=pd.DataFrame(res,columns=['b','ym','ssr','slope']).sort_values('ssr'); print(r.head(6).to_string())
# yearly means equal-weight months excl Nov
h['y']=h.ym.str[:4]; print(h.groupby('y').S.mean().round(4).to_dict()); print(h.groupby('y').C.mean().round(4).to_dict())
