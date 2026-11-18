# Extracted verbatim from the session log 2026-11-05T11:07:15.500675Z (inline code run in /home/claude/sina).
# Inputs (price JSON, Eastmoney metadata, corpus) are not in this repository.
# 2026-11-05T11:07:15.500675Z
python3 -I - <<'EOF' 2>&1 | grep -v -i "prefix\|loading\|dumping\|dict"
import json, re, pandas as pd, numpy as np, jieba
# 把东财预测(2024/2025窗口)和新浪摘要得分按 代码+日期+券商 对接
f=pd.read_csv('em_window_forecasts.csv',dtype={'stockCode':str}); f['stockCode']=f.stockCode.str.zfill(6); f['date']=pd.to_datetime(f.date).dt.strftime('%Y-%m-%d')
f['orgShort']=f.orgSName.str.replace('证券','',regex=False)
lab=pd.read_csv('out_pilot/candidate_words_labeled_v1.csv'); S=set(lab[lab.label.str.startswith('S')].word)
NEG=set("下滑 下降 亏损 减少 放缓 疲软 低迷 恶化 收缩 转亏 不及".split()); CJK=re.compile(r'^[一-鿿]{2,}$')
rows=[]
for y in ['2024','2025']:
    for line in open(f'base/{y}.jsonl',encoding='utf-8'):
        d=json.loads(line)
        if d['rtype']!='公司' or len(d['text'])<80: continue
        m=re.search(r'[（(](\d{6})[）)]',d['title'])
        if not m: continue
        t=[w for w in jieba.cut(d['text']) if CJK.match(w)]; n=max(len(t),1)
        rows.append(dict(stockCode=m.group(1),date=d['date'],orgShort=re.sub(r'(股份|有限|责任|公司|研究所|证券)+$','',d['org']),s_score=sum(w in S for w in t)/n,neg=sum(w in NEG for w in t)/n,ntok=n,rptid=d['rptid']))
s=pd.DataFrame(rows).drop_duplicates(['stockCode','date','orgShort'])
j=f.merge(s,on=['stockCode','date','orgShort'],how='inner')
print('东财预测', len(f[f.year.isin([2024,2025])]), '→ 匹配到新浪摘要', len(j), j.groupby('year').size().to_dict())
j.to_csv('forecast_matched.csv',index=False)
EOF
