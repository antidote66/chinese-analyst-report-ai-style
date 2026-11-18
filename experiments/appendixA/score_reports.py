import json, re, sys, pandas as pd, jieba
y=sys.argv[1]
lab = pd.read_csv('out_pilot/candidate_words_labeled_v1.csv'); S=set(lab[lab.label.str.startswith('S')].word)
PURE=set("扰动 兑现 韧性 验证 协同 节奏 叠加 增量 支撑 延续 释放 落地 优化 布局 推进 拓展 深化 强化 夯实 聚焦 赋能 深耕 共振 动能 中枢 闭环 迭代 适配".split())
EUPH=set("承压 修复 扰动 阶段性 短期 有望 静待 拐点 企稳 蓄势 韧性 改善 回暖 向好 筑底".split())
NEG=set("下滑 下降 亏损 减少 放缓 疲软 低迷 恶化 收缩 转亏 不及".split())
CJK=re.compile(r'^[一-鿿]{2,}$'); rows=[]
for line in open(f'base/{y}.jsonl',encoding='utf-8'):
    d=json.loads(line)
    if d['rtype']!='公司' or len(d['text'])<80: continue
    m=re.search(r'[（(](\d{6})[）)]', d['title'])
    if not m: continue
    t=[w for w in jieba.cut(d['text']) if CJK.match(w)]; n=len(t) or 1
    rows.append(dict(rptid=d['rptid'],date=d['date'],code=m.group(1),org=d['org'],ntok=n,s_score=sum(w in S for w in t)/n,pure=sum(w in PURE for w in t)/n,euph=sum(w in EUPH for w in t)/n,neg=sum(w in NEG for w in t)/n))
df=pd.DataFrame(rows); df.to_csv(f'rep{y}.csv',index=False)
open(f'/mnt/user-data/outputs/codes{y}.txt','w').write('\n'.join(['000300']+sorted(df.code.unique())))
print(y, len(df), df.code.nunique())
