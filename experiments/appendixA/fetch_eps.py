import requests, json, time, os
codes=[c.strip() for c in open('eps_codes.txt') if c.strip()]
out='eps_actual.json'; eps=json.load(open(out)) if os.path.exists(out) else {}
s=requests.Session()
for c in codes:
    if c in eps: continue
    try:
        r=s.get('https://datacenter-web.eastmoney.com/api/data/v1/get',params=dict(reportName='RPT_LICO_FN_CPD',columns='SECURITY_CODE,REPORTDATE,BASIC_EPS,NOTICE_DATE',filter=f'(SECURITY_CODE="{c}")',pageSize=40,sortColumns='REPORTDATE',sortTypes=-1),headers={'User-Agent':'Mozilla/5.0'},timeout=20).json()
        d=(r.get('result') or {}).get('data') or []
        eps[c]={x['REPORTDATE'][:10]:{'eps':x['BASIC_EPS'],'notice':(x.get('NOTICE_DATE') or '')[:10]} for x in d}
    except Exception as e:
        eps[c]={'_err':str(e)[:80]}
    if len(eps)%100==0: json.dump(eps,open(out,'w'))
    time.sleep(0.2)
json.dump(eps,open(out,'w')); print('done',len(eps))
