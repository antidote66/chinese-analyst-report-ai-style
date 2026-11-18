# Extracted verbatim from the session log 2026-11-05T11:35:20.565146Z (inline code run in /home/claude/sina).
# Inputs (price JSON, Eastmoney metadata, corpus) are not in this repository.
import json, pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
eps=json.load(open('eps_actual.json')); ind=json.load(open('industry.json'))
j=pd.read_csv('forecast_matched.csv',dtype={'stockCode':str})
j['actual']=[ (eps.get(c,{}).get(f'{y}-12-31') or {}).get('eps') for c,y in zip(j.stockCode,j.year)]
j['ind']=j.stockCode.map(lambda c:(ind.get(c) or {}).get('ind2'))
j=j[j.actual.notna()&(j.actual.abs()>0.02)&j.ind.notna()].copy()
j['err']=(j.eps-j.actual).abs()/j.actual.abs(); j['lerr']=np.log1p(j.err); j['bias']=(j.eps-j.actual)/j.actual.abs(); j['lbias']=np.sign(j.bias)*np.log1p(j.bias.abs())
for c in ['s_score','neg','ntok']: j['z_'+c]=((np.log(j[c]) if c=='ntok' else j[c])-(np.log(j[c]) if c=='ntok' else j[c]).mean())/(np.log(j[c]) if c=='ntok' else j[c]).std()
j['sy']=j.stockCode+'_'+j.year.astype(str); multi=j.groupby('sy').sy.transform('size')>=2
print('样本',len(j),'| 预测高于实际的比例', round((j.bias>0).mean(),3),'| 中位偏误',round(j.bias.median(),3),'| 同股-年≥2家',int(multi.sum()))
for yv,lab in [('lerr','|误差| (log1p)'),('lbias','乐观偏误 (signed log1p)')]:
    m1=smf.ols(f'{yv} ~ z_s_score + z_neg + z_ntok + C(year) + C(ind) + C(orgSName)',data=j).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(j.stockCode)[0]})
    d2=j[multi]; m2=smf.ols(f'{yv} ~ z_s_score + z_neg + z_ntok + C(sy)',data=d2).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(d2.stockCode)[0]})
    print(f'{lab:26s} 行业+年+券商FE: 风格 {m1.params["z_s_score"]:+.4f} (t={m1.tvalues["z_s_score"]:.2f}) 负面词 {m1.params["z_neg"]:+.4f} (t={m1.tvalues["z_neg"]:.2f}) | 同股-年内: 风格 {m2.params["z_s_score"]:+.4f} (t={m2.tvalues["z_s_score"]:.2f}) 负面词 {m2.params["z_neg"]:+.4f} (t={m2.tvalues["z_neg"]:.2f})')
q=pd.qcut(j.s_score.rank(method='first'),5,labels=[1,2,3,4,5])
print('\n风格密度五分位: 中位|误差| / 中位偏误 / 高估比例'); print(j.groupby(q).agg(err=('err','median'),bias=('bias','median'),over=('bias',lambda x:(x>0).mean())).round(3).to_string())
b=j.groupby(['orgSName','year']).agg(n=('err','size'),s=('s_score','mean'),err=('err','median'),bias=('bias','median')).reset_index(); b=b[b.n>=15]
print('\n券商-年层面 (≥15篇, %d个): 风格密度 vs 中位误差 相关 %.3f ; vs 中位偏误 %.3f'%(len(b),b[['s','err']].corr().iloc[0,1],b[['s','bias']].corr().iloc[0,1]))
print(b.sort_values('s',ascending=False)[['orgSName','year','n','s','err','bias']].round(3).to_string(index=False))
j.to_csv('out_pilot/forecast_accuracy.csv',index=False); b.to_csv('out_pilot/forecast_broker_year.csv',index=False)
