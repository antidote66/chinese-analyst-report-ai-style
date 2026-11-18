# Extracted verbatim from the session log 2026-11-06T09:12:55.143071Z (inline code run in /home/claude/sina).
# Inputs (price JSON, Eastmoney metadata, corpus) are not in this repository.
import requests,pandas as pd
s=requests.Session(); rows=[]
for p in range(1,6):
    rows+=s.get('https://reportapi.eastmoney.com/report/list',params=dict(industryCode='*',pageSize=100,industry='*',rating='',ratingChange='',beginTime='2025-04-01',endTime='2025-04-20',pageNo=p,qType=0,orgCode='',code='*',rcode=''),timeout=30).json()['data']
d=pd.DataFrame(rows)
for c in ['predictThisYearEps','predictNextYearEps']: d[c]=pd.to_numeric(d[c],errors='coerce')
act={}
for c in d.stockCode.unique()[:250]:
    try:
        r=s.get('https://datacenter-web.eastmoney.com/api/data/v1/get',params=dict(reportName='RPT_LICO_FN_CPD',columns='SECURITY_CODE,REPORTDATE,BASIC_EPS',filter=f'(SECURITY_CODE="{c}")',pageSize=12,sortColumns='REPORTDATE',sortTypes=-1),timeout=20).json()['result']['data']
        act[c]={x['REPORTDATE'][:10]:x['BASIC_EPS'] for x in r}
    except: pass
o=[(r.predictThisYearEps,r.predictNextYearEps,act[r.stockCode].get('2024-12-31'),act[r.stockCode].get('2025-12-31'),act[r.stockCode].get('2026-12-31')) for _,r in d[d.stockCode.isin(act)].iterrows()]
t=pd.DataFrame(o,columns=['this','next','a24','a25','a26'])
print(t.notna().sum().to_dict())
for f,a in [('this','a25'),('this','a24'),('next','a25')]:
    x=t.dropna(subset=[f,a]); x=x[(x[a]>0.05)&(x[f]>0)]
    print(f,a,len(x),round((x[f]/x[a]).median(),3),round((x[f]>x[a]).mean(),3))
