#!/usr/bin/env python3
"""用 DeepSeek 为 2019 年研报骨架生成两类 AI 样本：gen（按事实生成）/ polish（润色人类原文）。断点续跑，写 data/gen/<mode>.jsonl"""
import json, sys, time, os, requests, concurrent.futures as cf
key=open('.env.deepseek').read().split('=')[1].strip()
MODE=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 1500; MODEL=sys.argv[3] if len(sys.argv)>3 else 'deepseek-chat'
sk=json.load(open('data/gen_skeletons_2019.json',encoding='utf-8'))[:N]
TAG=sys.argv[4] if len(sys.argv)>4 else ''
out=f'data/gen/{MODE}{TAG}.jsonl'; done=set()
if os.path.exists(out):
    done={json.loads(l)['rptid'] for l in open(out,encoding='utf-8')}
def prompt(d):
    if MODE=='gen':
        return ("你是一名券商行业分析师。根据以下事实，撰写一段400–600字的研报摘要，面向机构投资者，用简体中文，只输出摘要正文。\n"
                f"【公司】{d['name']}（{d['code']}）\n【发布日期】{d['date']}\n【事实】\n"+"\n".join('- '+f for f in d['facts'])+
                (("\n【评级】"+d['rating'][0]) if d['rating'] else ''))
    elif MODE=='rewriteN':
        return ("你是一名券商分析师。请改写下面这段研报摘要：可以调整结构和表述，全部事实、数字和结论保持不变，长度相近，用简体中文，只输出改写后的正文。\n\n"+d['text'])
    elif MODE=='rewrite':
        return ("你是一名资深券商分析师。请把下面这段研报摘要深度改写：可以调整结构、重组段落、替换表述，使其更有洞察和说服力，但全部事实、数字和结论保持不变，长度相近，用简体中文，只输出改写后的正文。\n\n"+d['text'])
    else:
        return ("你是一名券商研究所的编辑。请润色下面这段研报摘要，使其更流畅、专业，保持全部事实、数字和结论不变，长度相近，用简体中文，只输出润色后的正文。\n\n"+d['text'])
def call(d):
    for t in range(4):
        try:
            r=requests.post('https://api.deepseek.com/chat/completions',headers={'Authorization':f'Bearer {key}'},
                json={'model':MODEL,'messages':[{'role':'user','content':prompt(d)}],'temperature':0.7,'max_tokens':3000},timeout=120)
            j=r.json(); txt=j['choices'][0]['message']['content'].strip()
            return dict(rptid=d['rptid'],code=d['code'],name=d['name'],org=d['org'],date=d['date'],mode=MODE,model=j.get('model',MODEL),text=txt,usage=j.get('usage',{}))
        except Exception as e:
            time.sleep(2**t)
    return None
todo=[d for d in sk if d['rptid'] not in done]
print(MODE,'todo',len(todo),'done',len(done),flush=True)
with open(out,'a',encoding='utf-8') as f, cf.ThreadPoolExecutor(8) as ex:
    for i,res in enumerate(ex.map(call,todo)):
        if res: f.write(json.dumps(res,ensure_ascii=False)+'\n')
        if i%50==0: f.flush(); print(i,flush=True)
print('finished',MODE,flush=True)
