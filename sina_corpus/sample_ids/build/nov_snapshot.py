import json,re,sys,collections,pandas as pd,jieba
jieba.setLogLevel(60)
B='/sessions/'+__import__('os').environ['USER']+'/w/B/sina/' if False else sys.argv[1]
snap=pd.read_csv(sys.argv[2],encoding='utf-8-sig'); target=dict(zip(snap.year.astype(str),snap.n))
lab=pd.read_csv(B+'out_pilot/candidate_words_labeled_v1.csv')
S=lab[lab.label.str.startswith('S')].word.tolist(); E=lab[lab.label.str.startswith('E')].word.tolist()
CYC="下滑 下降 亏损 减少 放缓 疲软 低迷 恶化 收缩 回落 转亏".split()
CJK=re.compile(r"^[一-鿿]{2,}$")
seen=set(); docs=[]; taken=collections.Counter()
files=[B+'mac/2026.jsonl',B+'data/2026.jsonl']+[B+f'base/{y}.jsonl' for y in range(2019,2026)]
for f in files:
    for line in open(f,encoding='utf-8'):
        try:d=json.loads(line)
        except: continue
        if d['rptid'] in seen or not d.get('text') or len(d['text'])<80: continue
        md=d['date'][5:]
        if not('11-01'<=md<='11-04'): continue
        y=d['date'][:4]
        if taken[y]>=target.get(y,0): continue   # prefix truncation = 09:57 snapshot
        seen.add(d['rptid']); docs.append(d); taken[y]+=1
print(dict(taken),sum(taken.values()))
cnt=collections.defaultdict(collections.Counter); n=collections.Counter()
for d in docs:
    y=d['date'][:4]; n[y]+=1; cnt[y].update({w for w in jieba.cut(d['text']) if CJK.match(w)})
for y in sorted(n):
    s=sum(cnt[y][w] for w in S)/(n[y]*len(S)); c=sum(cnt[y][w] for w in CYC)/(n[y]*len(CYC))
    r=snap[snap.year.astype(str)==y].iloc[0]
    print(y,n[y],round(s,6),round(r.S_style,6),round(c,6),round(r.CYC_cycle,6),'match' if abs(s-r.S_style)<1e-9 and abs(c-r.CYC_cycle)<1e-9 else 'DIFF')
pd.DataFrame([dict(rptid=d['rptid'],date=d['date'],year=d['date'][:4],org=d['org'],rtype=d['rtype'],text_chars=len(d['text'])) for d in docs]).to_csv(sys.argv[3],index=False,encoding='utf-8')
