# Extracted verbatim from the session log 2026-11-06T09:12:55.143071Z (inline code run in /home/claude/sina).
# Inputs (price JSON, Eastmoney metadata, corpus) are not in this repository.
import pandas as pd, requests, numpy as np
d=pd.read_csv('out_pilot/forecast_accuracy.csv',dtype={'stockCode':str})
d['stockCode']=d.stockCode.str.zfill(6)
codes=d.stockCode.drop_duplicates().sample(250,random_state=1)
s=requests.Session(); act={}
for c in codes:
    try:
        r=s.get('https://datacenter-web.eastmoney.com/api/data/v1/get',params=dict(reportName='RPT_LICO_FN_CPD',columns='SECURITY_CODE,REPORTDATE,BASIC_EPS',filter=f'(SECURITY_CODE="{c}")',pageSize=40,sortColumns='REPORTDATE',sortTypes=-1),timeout=20).json()['result']['data']
        act[c]={x['REPORTDATE'][:10]:x['BASIC_EPS'] for x in r}
    except Exception as e: pass
rows=[]
for _,r in d[d.stockCode.isin(act)].iterrows():
    a=act[r.stockCode]; y=int(r.year)
    rows.append((r.year,r.eps,a.get(f'{y-1}-12-31'),a.get(f'{y}-12-31'),a.get(f'{y+1}-12-31')))
t=pd.DataFrame(rows,columns=['y','f','aPrev','aY','aNext']).dropna(subset=['f','aY'])
t=t[(t.aY>0.05)&(t.f>0)]
for col in ['aPrev','aY','aNext']:
    x=t.dropna(subset=[col]); x=x[x[col]>0.05]
    print(col,len(x),'median f/a',round((x.f/x[col]).median(),3),'share f>a',round((x.f>x[col]).mean(),3),'median |f-a|/a',round(((x.f-x[col]).abs()/x[col]).median(),3))
print(t.groupby('y').apply(lambda g: pd.Series({'n':len(g),'med f/aY':(g.f/g.aY).median(),'share>':(g.f>g.aY).mean()})))
