import json,glob,os,sys,collections,pandas as pd,jieba
jieba.setLogLevel(60)
B=sys.argv[1]; OUT=sys.argv[2]; A=sys.argv[3]
W=pd.read_csv(B+'out_pilot/candidate_words_labeled_v1.csv',encoding='utf-8-sig')
S=set(w for w,l in zip(W.word,W.label) if str(l).startswith('S')); cyc=set('下滑 下降 亏损 减少 放缓 疲软 低迷 恶化 收缩 回落 转亏'.split())
files=sorted(glob.glob(B+'panel/*.jsonl'))
for y in range(2019,2027):
    for f in [B+f'base/{y}.jsonl',B+f'data/{y}.jsonl']:
        if os.path.exists(f): files.append(f)
seen=set(); rows=[]
for f in files:
    for l in open(f,encoding='utf-8'):
        try: r=json.loads(l)
        except: continue
        d=r.get('date','')
        if r['rptid'] in seen or len(d)<10: continue
        if int(d[8:10])>4: continue
        seen.add(r['rptid']); t=r.get('text','')
        if len(t)<200: continue
        rows.append(dict(rptid=r['rptid'],date=d,ym=d[:7],org=r['org'],rtype=r['rtype'],source_file=os.path.relpath(f,B),text=t))
print('reports in windows',len(rows), 'windows',len(set(x['ym'] for x in rows)))
ref=pd.read_csv(A+'final/panel_reports.csv',dtype={'rptid':str})
print('panel_reports.csv rows',len(ref),'same ids/order:',list(ref.rptid)==[x['rptid'] for x in rows])
# recompute S/C for every report and compare with stored values
import numpy as np
bad=0
for x,(_,r) in zip(rows,ref.iterrows()):
    s=set(w for w in jieba.lcut(x['text']) if len(w)>=2)
    if abs(len(s&S)/len(S)-r.S)>1e-9 or abs(len(s&cyc)/len(cyc)-r.C)>1e-9: bad+=1
print('S/C recompute mismatches',bad)
pd.DataFrame([{k:v for k,v in x.items() if k!='text'}|{'text_chars':len(x['text'])} for x in rows]).to_csv(OUT+'/panel_ids.csv',index=False)
