import json,random,torch,numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score
torch.set_num_threads(6); random.seed(0)
gen=[json.loads(l) for l in open('data/gen/gen.jsonl',encoding='utf-8')][:1500]
pol=[json.loads(l) for l in open('data/gen/polish.jsonl',encoding='utf-8')][:1500]
hum=json.load(open('data/human_neg.json',encoding='utf-8'))[:len(gen)+len(pol)]
ids=sorted({d['rptid'] for d in gen+pol+hum}); random.shuffle(ids); n=len(ids)
split={i:('test' if k<n*0.15 else 'val' if k<n*0.25 else 'train') for k,i in enumerate(ids)}
tok=AutoTokenizer.from_pretrained('ckpt'); m=AutoModelForSequenceClassification.from_pretrained('ckpt').eval()
def pr(texts):
    out=[]
    with torch.no_grad():
        for i in range(0,len(texts),32):
            e=tok(texts[i:i+32],padding=True,truncation=True,max_length=320,return_tensors='pt'); out+=torch.softmax(m(**e).logits,-1)[:,1].tolist()
    return np.array(out)
G='/home/claude/gen/data/gen/'
ph=pr([d['text'] for d in hum if split[d['rptid']]=='test'])
res={'人类 2019（测试集）':dict(n=len(ph),rate=float((ph>0.5).mean()),mean_p=float(ph.mean()))}
print(res,flush=True)
sets=[('DeepSeek','润色','polish'),('DeepSeek','深度改写','rewrite'),('DeepSeek','独立生成','gen'),('千问','润色','polish_qwen'),('千问','深度改写','rewrite_qwen'),('千问','独立生成','gen_qwen'),('Claude','润色','polish_claude'),('Claude','深度改写','rewrite_claude'),('Claude','独立生成','gen_claude')]
for mo,c,f in sets:
    rows=[json.loads(l) for l in open(G+f+'.jsonl',encoding='utf-8')]
    t=[r['text'] for r in rows if split.get(r['rptid'])=='test']   # only test-split skeletons (never seen in training)
    p=pr(t); y=np.r_[np.ones(len(p)),np.zeros(len(ph))]; auc=roc_auc_score(y,np.r_[p,ph])
    res[f'{mo} {c}']=dict(n=len(t),rate=float((p>0.5).mean()),mean_p=float(p.mean()),auc=float(auc)); print(mo,c,res[f'{mo} {c}'],flush=True)
json.dump(res,open('out/xmodel_eval2.json','w'),ensure_ascii=False,indent=1)
