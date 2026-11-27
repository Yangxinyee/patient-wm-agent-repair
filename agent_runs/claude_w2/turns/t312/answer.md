fwd_9 gave a value NLL of 1.57204 for the current edit (cand3: dose numerics ×5 plus a gated insulin readout in the value head) on 100 tuning stays. On its own this number tells me nothing. If these are the same sel1 stays where the original scored 1.52793, then cand3 is about 0.044 nats worse. That would fail the 0.005 budget badly, and I would have to scale it back or revert to the safe numeric-only edit. That edit was ×3, which improved factual NLL and moved NE ×2 from 1.08 to 2.66.

I have only 7 Python runs left, so this run does all the diagnosis at once:
- which stays fwd_9 used, and the original NLL on those same stays, by event;
- summaries of earlier passes;
- exactly what each current edit changed;
- the APIs of my local re-implementation and the tail of the script that built cand3.

Confirming outcome: fwd_9 used sel1, and the per-event NLL shows that glucose (the head readout) is the source of the loss, so cand3 is over budget and must be scaled down. Refuting outcome: the stays differ, or the original NLL on them is close to 1.572, so cand3 may already fit and the next step is a dose_edit pass on it.

```python
import os, json, glob, time
import numpy as np, pandas as pd
Wd='/work'
def P(*a): print(*a, flush=True)
for k in range(1,10):
    d=f'{Wd}/fwd_{k}'
    try:
        fl=sorted(os.listdir(d)); P('fwd',k,fl)
        for fn in fl:
            if fn.endswith(('.json','.txt','.md')):
                s=open(f'{d}/{fn}').read(); P('   ',fn,s[:300].replace('\n',' '))
    except Exception as ex: P('ERR fwd',k,ex)
def keycols(df):
    sid=next(c for c in df.columns if 'stay' in c.lower())
    ev=next((c for c in df.columns if c.lower() in ('event','code','event_type','event_name','name','kind','variable')), None)
    nl=[c for c in df.columns if 'nll' in c.lower()]
    return sid,ev,nl
s9=[]
try:
    f9=pd.read_csv(f'{Wd}/fwd_9/factual_edited.csv')
    P('f9',f9.shape,f9.columns.tolist()); P(f9.head(2).to_string())
    sid,ev,nl=keycols(f9); P('keys',sid,ev,nl, 'mean nll',f9[nl[0]].mean())
    s9=sorted(f9[sid].unique().tolist())
except Exception as ex: P('ERR f9',ex)
summ=[]
try:
    for fn in sorted(glob.glob(f'{Wd}/fwd_*/factual_*.csv')):
        df=pd.read_csv(fn); sd,e2,n2=keycols(df)
        s=set(df[sd].unique()); ov=len(s & set(s9)); d=df[df[sd].isin(s9)]
        if len(d):
            m=(fn.split('/')[-2:], 'n',len(s),'ov',ov,'rows',len(d),'nll %.5f'%d[n2[0]].mean(), {k:round(v,4) for k,v in d.groupby(e2)[n2[0]].mean().items()})
        else: m=(fn.split('/')[-2:], 'n',len(s),'ov',0,'nll(all) %.5f'%df[n2[0]].mean())
        P(*m); summ.append(str(m))
except Exception as ex: P('ERR cmp',ex)
for nm in ['sel1.json','selT.json']:
    try:
        j=json.load(open(f'{Wd}/{nm}'))
        if isinstance(j,dict):
            P(nm,'dict',{k:(len(v) if hasattr(v,'__len__') else v) for k,v in j.items()})
        else: P(nm,len(j),j[:3],'ov f9',len(set(map(str,j))&set(map(str,s9))))
    except Exception as ex: P('ERR sel',nm,ex)
P('S9', json.dumps([int(x) if isinstance(x,(np.integer,int)) else x for x in s9]))
try:
    for fn in sorted(glob.glob(f'{Wd}/fwd_*/dose_edit_*.csv')):
        df=pd.read_csv(fn)
        ec=next((c for c in df.columns if 'edit' in c.lower() or 'arm' in c.lower()),None)
        sc=[c for c in df.columns if 'shift' in c.lower()]
        sd=next((c for c in df.columns if 'stay' in c.lower()),None)
        msg=f'{fn.split("/")[-2:]} {df.shape} cols={df.columns.tolist()[:10]}'
        if ec and sc: msg+=' '+str(df.groupby(ec)[sc[0]].mean().round(3).to_dict())
        if sd: msg+=f' nst={df[sd].nunique()} ov9={len(set(df[sd].unique())&set(s9))}'
        P(msg)
except Exception as ex: P('ERR de',ex)
try:
    for n in edited():
        a=np.asarray(load(n),dtype=np.float64); b=np.asarray(load_original(n),dtype=np.float64); d=a-b
        if d.ndim==2:
            rows=np.where(np.abs(d).sum(1)>1e-6)[0]; cs=np.where(np.abs(d).sum(0)>1e-6)[0]
            P(n,a.shape,'rel %.4f'%(np.linalg.norm(d)/np.linalg.norm(b)),'rows',rows[:12].tolist(),len(rows),'cols',cs[:12].tolist(),len(cs))
        else:
            idx=np.where(np.abs(d)>1e-7)[0]; P(n,a.shape,'idx',idx[:12].tolist(),np.round(d[idx[:5]],3))
    a=np.asarray(load('embed.numeric.weight'),dtype=np.float64); b=np.asarray(load_original('embed.numeric.weight'),dtype=np.float64)
    if a.shape[1]<=16: P('numeric col ratios',[(j,round(float(a[:,j]@b[:,j]/(b[:,j]@b[:,j]+1e-12)),3)) for j in range(a.shape[1])])
    else: P('numeric row ratios',[(j,round(float(a[j]@b[j]/(b[j]@b[j]+1e-12)),3)) for j in range(a.shape[0])])
except Exception as ex: P('ERR edits',ex)
try:
    pys=sorted(glob.glob(f'{Wd}/*.py'), key=os.path.getmtime)
    for fn in pys:
        src=open(fn).read().splitlines()
        defs=[l[4:l.find(')')+1] for l in src if l.startswith('def ')]
        P(os.path.basename(fn), time.strftime('%H:%M',time.localtime(os.path.getmtime(fn))), len(src), defs)
    P('--- tail of', os.path.basename(pys[-1])); P('\n'.join(open(pys[-1]).read().splitlines()[-45:]))
except Exception as ex: P('ERR py',ex)
try:
    info=[]
    for k,v in dict(globals()).items():
        if k.startswith('_') or k in ('In','Out','exit','quit','get_ipython'): continue
        if isinstance(v,np.ndarray): info.append(f'{k}{list(v.shape)}')
        elif isinstance(v,pd.DataFrame): info.append(f'{k}:df{list(v.shape)}')
        elif isinstance(v,dict): info.append(f'{k}:d{len(v)}')
        elif isinstance(v,(list,tuple)): info.append(f'{k}:l{len(v)}')
        elif isinstance(v,float): info.append(f'{k}={v:.4g}')
        elif isinstance(v,int) and not isinstance(v,bool): info.append(f'{k}={v}')
        elif callable(v) and getattr(v,'__module__','')=='__main__' and not isinstance(v,type): info.append(f'{k}()')
    P(' '.join(info)[:5000])
except Exception as ex: P('ERR glob',ex)
with open(f'{Wd}/notes.md','a') as f:
    f.write('\n## Restart 8 (run 294): diagnosis of fwd_9 (cand3 factual, edited only, 1.57204)\n'+'\n'.join('- '+s for s in summ)+'\n')
```