import json,random,torch,numpy as np,pandas as pd,sys
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f: matplotlib.rcParams['font.family']=font_manager.FontProperties(fname=f).get_name();break
torch.set_num_threads(2);random.seed(2)
G='/home/claude/gen2/'
tok=AutoTokenizer.from_pretrained(G+'ckpt');m=AutoModelForSequenceClassification.from_pretrained(G+'ckpt').eval()
excl={json.loads(l)['rptid'] for f in ('gen','polish') for l in open(G+f'data/gen/{f}.jsonl')}|{d['rptid'] for d in json.load(open(G+'data/human_neg.json'))}
def sc(ds):
    o=[]
    with torch.no_grad():
        for i in range(0,len(ds),16):
            e=tok([d['text'] for d in ds[i:i+16]],padding=True,truncation=True,max_length=320,return_tensors='pt')
            o+=torch.softmax(m(**e).logits,-1)[:,1].tolist()
    return np.array(o)
rows=[]
for y in range(2019,2027):
    srcs=['/home/claude/sina/mac/2026.jsonl','/home/claude/sina/data/2026.jsonl'] if y==2026 else [f'/home/claude/sina/base/{y}.jsonl']
    pool=[];seen=set()
    for s in srcs:
        for l in open(s):
            d=json.loads(l)
            if d['rptid'] in seen: continue
            seen.add(d['rptid'])
            if d['rtype']=='公司' and 400<=len(d['text'])<=2000 and '10-01'<=d['date'][5:]<='11-04' and d['rptid'] not in excl: pool.append(d)
    random.shuffle(pool);pool=pool[:400];p=sc(pool)
    rows.append(dict(year=y,n=len(pool),mean_p=p.mean(),rate=(p>.5).mean()));print(rows[-1],flush=True)
df=pd.DataFrame(rows);df.to_csv('/home/claude/final/detector_by_year_seasonal.csv',index=False)
words=pd.read_csv('/home/claude/sina/out_curves/year_curves.csv') if False else None
sty={2019:0.113,2020:0.131,2021:0.127,2022:0.134,2023:0.139,2024:0.138,2025:0.152,2026:0.174}
fig,ax=plt.subplots(figsize=(7,4));ax2=ax.twinx()
ax.plot(df.year,[sty[y] for y in df.year],'o-',label='风格词组文档频率(左)');ax2.plot(df.year,df.mean_p,'s--',c='C3',label='检测器平均AI概率(右)')
ax.set_xlabel('年份(10月1日–11月4日窗口)');ax.legend(loc='upper left');ax2.legend(loc='lower right');fig.tight_layout();fig.savefig('/home/claude/final/fig_detector_vs_words.png',dpi=160)
