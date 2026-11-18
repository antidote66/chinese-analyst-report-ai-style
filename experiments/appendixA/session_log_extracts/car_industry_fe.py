# Extracted verbatim from the session log 2026-11-05T10:27:54.438464Z (inline code run in /home/claude/sina).
# Inputs (price JSON, Eastmoney metadata, corpus) are not in this repository.
import json, pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
ind=json.load(open('industry.json'))
res=[]
for y in ['2023','2024','2025']:
    rep=pd.read_csv(f'rep{y}_car.csv',dtype={'code':str})
    rep['ind']=rep.code.map(lambda c: (ind.get(c) or {}).get('ind2')); rep['lnmcap']=rep.code.map(lambda c: np.log((ind.get(c) or {}).get('mcap') or np.nan))
    rep=rep.dropna(subset=['ind','lnmcap','pre20','lnamt'])
    print(f'===== {y}  n={len(rep)}  行业数={rep.ind.nunique()}   加入：行业FE + 市值')
    for yv in ['car5','car20','car60']:
        d=rep.dropna(subset=[yv]); out=[]
        for spec in ['z_s_score','z_euph','z_pure','z_neg']:
            m=smf.ols(f'{yv} ~ {spec} + z_neg*0 + pre20 + lnamt + lnmcap + np.log(n_rep) + C(org) + C(date) + C(ind)' if spec=='z_neg' else f'{yv} ~ {spec} + z_neg + pre20 + lnamt + lnmcap + np.log(n_rep) + C(org) + C(date) + C(ind)', data=d).fit(cov_type='cluster', cov_kwds={'groups':pd.factorize(d.code)[0]})
            out.append(f'{spec[2:]}: {m.params[spec]*100:+.2f}% (t={m.tvalues[spec]:.2f})'); res.append((y,yv,spec[2:],m.params[spec]*100,m.tvalues[spec]))
        print(f'  {yv:6s} '+'  '.join(out))
pd.DataFrame(res,columns=['year','horizon','score','coef_pct','t']).to_csv('out_pilot/car_regressions.csv',index=False)
