# Extracted verbatim from the session log 2026-11-05T10:27:54.438464Z (inline code run in /home/claude/sina).
# Inputs (price JSON, Eastmoney metadata, corpus) are not in this repository.
import json, pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
ind=json.load(open('industry.json'))
print('H1 信息含量：研报发布后的反应幅度 |CAR| 是否随风格密度变化（行业FE+市值+控制）')
for y in ['2023','2024','2025']:
    rep=pd.read_csv(f'rep{y}_car.csv',dtype={'code':str})
    rep['ind']=rep.code.map(lambda c:(ind.get(c) or {}).get('ind2')); rep['lnmcap']=rep.code.map(lambda c: np.log((ind.get(c) or {}).get('mcap') or np.nan))
    rep=rep.dropna(subset=['ind','lnmcap','pre20','lnamt','car5','car20'])
    rep['abs5']=rep.car5.abs(); rep['abs20']=rep.car20.abs(); rep['z_len']=(np.log(rep.ntok)-np.log(rep.ntok).mean())/np.log(rep.ntok).std()
    out=[]
    for yv in ['abs5','abs20']:
        for spec in ['z_s_score','z_pure','z_euph']:
            m=smf.ols(f'{yv} ~ {spec} + z_neg + z_len + pre20 + lnamt + lnmcap + np.log(n_rep) + C(org) + C(date) + C(ind)', data=rep).fit(cov_type='cluster', cov_kwds={'groups':pd.factorize(rep.code)[0]})
            out.append(f'{yv}/{spec[2:]}: {m.params[spec]*100:+.2f}% (t={m.tvalues[spec]:.2f})')
    print(f'  {y} n={len(rep)}  '+'  '.join(out))
