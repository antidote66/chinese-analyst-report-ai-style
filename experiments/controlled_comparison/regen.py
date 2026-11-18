import json,sys,time,requests,concurrent.futures as cf,importlib.util
prov,mode,fname=sys.argv[1],sys.argv[2],sys.argv[3]
sk={d['rptid']:d for d in json.load(open('data/gen_skeletons_2019.json',encoding='utf-8'))}
src=open('gen_samples.py',encoding='utf-8').read()
P=src[src.index('def prompt(d):'):src.index('def call(d):')]
MODE=mode; exec(P)
if prov=='ds': url='https://api.deepseek.com/chat/completions'; key=open('.env.deepseek').read().split('=',1)[1].strip(); model='deepseek-chat'
else: url='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'; key=open('.env.qwen').read().split('=',1)[1].strip(); model='qwen-plus'
rows=[json.loads(l) for l in open(fname,encoding='utf-8')]
bad=[i for i,r in enumerate(rows) if r.get('usage',{}).get('completion_tokens',0)>=895]
print(fname,'truncated',len(bad),flush=True)
def call(i):
    d=sk[rows[i]['rptid']]
    for t in range(5):
        try:
            j=requests.post(url,headers={'Authorization':f'Bearer {key}'},json={'model':model,'messages':[{'role':'user','content':prompt(d)}],'temperature':0.7,'max_tokens':3000},timeout=240).json()
            txt=j['choices'][0]['message']['content'].strip()
            return i,txt,j.get('usage',{}),j['choices'][0].get('finish_reason')
        except Exception: time.sleep(2**t)
    return i,None,None,None
with cf.ThreadPoolExecutor(8) as ex:
    for i,txt,u,fr in ex.map(call,bad):
        if txt: rows[i]['text']=txt; rows[i]['usage']=u; rows[i]['finish']=fr; rows[i]['regen_cap3000']=True
with open(fname,'w',encoding='utf-8') as f:
    for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
print('done',fname,sum(1 for r in rows if r.get('regen_cap3000')),flush=True)
