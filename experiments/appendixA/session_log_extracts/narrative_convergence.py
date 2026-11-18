# Extracted verbatim from the session log 2026-11-05T11:07:15.500675Z (inline code run in /home/claude/sina).
# Inputs (price JSON, Eastmoney metadata, corpus) are not in this repository.
# 2026-11-05T11:07:15.500675Z
python3 -I - <<'EOF' 2>&1 | grep -v -i "prefix\|loading\|dumping\|dict"
import json, re, numpy as np, pandas as pd, jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
CJK=re.compile(r'^[一-鿿]{2,}$')
docs=[]
for src in ['mac/2026.jsonl','data/2026.jsonl']+[f'base/{y}.jsonl' for y in range(2019,2026)]:
    for line in open(src,encoding='utf-8'):
        d=json.loads(line)
        if d['rtype']!='公司' or len(d['text'])<80: continue
        m=re.search(r'[（(](\d{6})[）)]',d['title'])
        if not m: continue
        if not ('10-01'<=d['date'][5:]<='11-04'): continue
        docs.append(dict(rptid=d['rptid'],year=d['date'][:4],code=m.group(1),org=d['org'],text=d['text']))
df=pd.DataFrame(docs).drop_duplicates('rptid').reset_index(drop=True)
df['toks']=[' '.join(w for w in jieba.cut(t) if CJK.match(w)) for t in df.text]
X=TfidfVectorizer(min_df=5, max_df=0.5, sublinear_tf=True).fit_transform(df.toks)
rows=[]
for (y,c),g in df.groupby(['year','code']):
    g=g.drop_duplicates('org')
    if len(g)<2: continue
    S=cosine_similarity(X[g.index.values]); iu=np.triu_indices(len(g),1)
    rows.append(dict(year=y,code=c,n_org=len(g),sim_mean=S[iu].mean()))
pair=pd.DataFrame(rows)
rng=np.random.default_rng(0); base=[]
for y,g in df.groupby('year'):
    idx=rng.choice(g.index.values, size=(2000,2))
    idx=idx[df.loc[idx[:,0],'code'].values!=df.loc[idx[:,1],'code'].values]
    s=[cosine_similarity(X[a],X[b])[0,0] for a,b in idx]
    base.append(dict(year=y, cross_stock_sim=float(np.mean(s))))
out=pair.groupby('year').agg(stocks=('code','size'), avg_brokers=('n_org','mean'), same_stock_sim=('sim_mean','mean')).reset_index().merge(pd.DataFrame(base),on='year')
out['excess_sim']=out.same_stock_sim-out.cross_stock_sim
print(out.round(4).to_string(index=False))
out.to_csv('out_pilot/convergence_by_year.csv',index=False); pair.to_csv('out_pilot/convergence_pairs.csv',index=False); df[['rptid','year','code','org']].to_csv('out_pilot/company_reports_index.csv',index=False)
EOF
