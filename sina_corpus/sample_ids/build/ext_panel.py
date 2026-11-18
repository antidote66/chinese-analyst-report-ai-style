import json,re,sys,collections,pandas as pd,jieba
jieba.setLogLevel(60)
B=sys.argv[1]; OUT=sys.argv[2]
CJK=re.compile(r"^[一-鿿]{2,}$")
# --- extended window (as in the 2026-11-05 lifecycle / HC3 run): per-file dedup, len>=80, 10-01..11-04
srcs=['mac/2026.jsonl','data/2026.jsonl']+[f'base/{y}.jsonl' for y in range(2019,2026)]
rows=[];n=collections.Counter();cnt=collections.defaultdict(collections.Counter)
for s in srcs:
    seen=set()
    for line in open(B+s,encoding='utf-8'):
        d=json.loads(line)
        if d['rptid'] in seen or len(d.get('text',''))<80 or not('10-01'<=d['date'][5:]<='11-04'): continue
        seen.add(d['rptid']); y=d['date'][:4]; n[y]+=1
        rows.append(dict(rptid=d['rptid'],date=d['date'],year=y,org=d['org'],rtype=d['rtype'],source_file=s,text_chars=len(d['text'])))
        cnt[y].update({w for w in jieba.cut(d['text']) if CJK.match(w)})
print('extended',dict(n),sum(n.values()))
ext=pd.DataFrame(rows); ext.to_csv(OUT+'/ext_ids.csv',index=False)
F=pd.read_csv(B+'out_pilot/doc_freq_by_year_full.csv',encoding='utf-8-sig',index_col=0)
bad=0
for w in ['承压','修复','下滑','扰动','韧性','算力','扣非','毛利率']:
    for y in sorted(n):
        v=cnt[y][w]/n[y]
        if abs(v-F.loc[w,y])>1e-9: bad+=1
print('doc_freq check mismatches:',bad)
# company subset (convergence / Appendix A): rtype 公司, 6-digit code in title, global drop_duplicates rptid
comp=[];
for s in srcs:
    for line in open(B+s,encoding='utf-8'):
        d=json.loads(line)
        if d['rtype']!='公司' or len(d['text'])<80: continue
        m=re.search(r'[（(](\d{6})[）)]',d['title'])
        if not m or not('10-01'<=d['date'][5:]<='11-04'): continue
        comp.append(dict(rptid=d['rptid'],date=d['date'],year=d['date'][:4],org=d['org'],code=m.group(1)))
c=pd.DataFrame(comp).drop_duplicates('rptid'); print('company',len(c),c.groupby('year').size().to_dict())
c.to_csv(OUT+'/ext_company_ids.csv',index=False)
