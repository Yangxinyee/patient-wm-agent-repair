Run 295 achieved two things and failed at a third:
- **Done:** the head-readout edits are reverted, so only `embed.numeric.weight` (dose and rate columns ×5) remains edited.
- **Learned:** the rate numerics are linear, `z = (x − median)/iqr` clipped to ±10. So the saturation of NE ×2 (2.66 at ×3 → 2.82 at ×5) happens downstream of the input, not in the feature itself.
- **Failed:** the local scan produced nothing usable. My rollout engine's NE key is `'ne'`, not `'nor'`, so all NE arms were copies of the recorded-dose arm. I also printed raw arrays instead of averaging them.

This run redoes the scan correctly, on all 100 sel1 stays and with 9 arms. The insulin arms are ×2, ×½ and ×0; the NE arms are ×2, ×½, ×0, ×3 and ×1.5. I also read NE ×2 on map_cuff and insulin ×2 on glucose_lab.

The numeric scale factors are (1,1), (3,3), (5,5), (7,7), (10,10), (5,10) and (10,5). The tool already measured (3,3) and (5,5), at 2.66/−1.48 and 2.82/−2.22, so those two validate the local engine. I also try my local teacher-forced NLL as a factual check.

The run then saves the factor with the smallest summed relative error against my NE effect estimates of +4.7, −2.35, −4.7, +9.4 and +2.35. Those estimates are linear in the dose change. I will verify the saved choice with the tool afterwards.

- **Confirms the plan:** local (5,5) gives about +2.8 / −2.2, and one factor is clearly better on the error.
- **Refutes it:** local numbers disagree with the tool, or every factor saturates NE ×2 near 3. In that case I keep ×5.

```python
import time, json, re, traceback
import numpy as np, pandas as pd, torch
Wd='/work'
def P(*a): print(*a, flush=True)
T0=time.time()
tun=pd.read_csv('/data/tuning.csv'); sel1=json.load(open(f'{Wd}/sel1.json'))
P(open(f'{Wd}/tf.py').read()[:900])
R=prep_rollout(tun, sel1)
is_t=isinstance(load('embed.numeric.weight'),torch.Tensor)
b=np.asarray(load_original('embed.numeric.weight'),dtype=np.float32)
def mk_num(k2,k4):
    a=b.copy(); a[:,2]*=k2; a[:,4]*=k4
    return torch.from_numpy(a) if is_t else a
def mk_loader(k2,k4):
    at=mk_num(k2,k4)
    return lambda n: at if n=='embed.numeric.weight' else load(n)
arms=[{}, {'ins':2.0}, {'ins':0.5}, {'ins':0.0}, {'ne':2.0}, {'ne':0.5}, {'ne':0.0}, {'ne':3.0}, {'ne':1.5}]
names=['f','i2','i.5','i0','n2','n.5','n0','n3','n1.5']
def agg(res,drug,outcome,aidx):
    rows,reached,out=readout(R,res,arms,drug=drug,outcome=outcome)
    A=R['A']; si=A.si.values[rows]; tt=A.t.values[rows]; r={}
    for a in aidx:
        d=np.nanmean(out[a]-out[0],axis=1); m=np.asarray(reached,bool)&np.isfinite(d)
        sh=pd.DataFrame({'si':si[m],'h':np.floor(tt[m]),'d':d[m]}).groupby(['si','h']).d.mean()
        r[names[a]]=round(float(sh.mean()),2)
    return r
TRUTH={'n2':4.7,'n.5':-2.35,'n0':-4.7,'n3':9.4,'n1.5':2.35}
def relerr(v): return round(sum(abs(v['ne'][k]-t)/abs(t) for k,t in TRUTH.items()),3)
scan={}
for (k2,k4) in [(1,1),(3,3),(5,5),(7,7),(10,10),(5,10),(10,5)]:
    if time.time()-T0>620: P('time guard'); break
    try:
        t1=time.time(); Wv=getW(mk_loader(k2,k4)); res=run_rollout(R,Wv,arms,n_paths=10,seed=0)
        v=dict(ins=agg(res,'insulin','glucose_finger',[1,2,3]), ne=agg(res,'norepinephrine','map_arterial',[4,5,6,7,8]),
               cuff=agg(res,'norepinephrine','map_cuff',[4]), lab=agg(res,'insulin','glucose_lab',[1]))
        v['relerr']=relerr(v); scan[(k2,k4)]=v; P((k2,k4),'%.0fs'%(time.time()-t1),v)
    except Exception as ex: P('ERR',(k2,k4),traceback.format_exc()[-800:]); break
json.dump({str(k):v for k,v in scan.items()}, open(f'{Wd}/scan296.json','w'))
def nll_of(r):
    if isinstance(r,pd.DataFrame):
        c=[c for c in r.columns if 'nll' in c.lower()]; return float(r[c[0]].mean()) if c else None
    if isinstance(r,dict):
        for k,v in r.items():
            if 'nll' in str(k).lower():
                try: return float(np.nanmean(np.asarray(v,dtype=float)))
                except Exception: pass
    if isinstance(r,(tuple,list)):
        for x in r:
            v=nll_of(x)
            if v is not None: return v
    return None
best=None
try:
    cands={k:v for k,v in scan.items() if k!=(1,1) and max(k)<=10}
    if cands:
        best=min(cands,key=lambda k:cands[k]['relerr']); P('BEST',best,cands[best])
        save_edit('embed.numeric.weight', mk_num(*best)); P('saved; edited', edited())
except Exception as ex: P('ERR save',traceback.format_exc()[-600:])
tfres={}
for nm,ld in [('orig',load_original),('best',load)]:
    if time.time()-T0>800: P('tf time guard'); break
    try:
        t1=time.time(); r=teacher_forced(tun, sel1, getW(ld))
        tfres[nm]=nll_of(r); P('TF',nm,tfres[nm],type(r).__name__,(r.columns.tolist() if isinstance(r,pd.DataFrame) else ''),'%.0fs'%(time.time()-t1))
    except Exception as ex: P('ERR tf',nm,traceback.format_exc()[-500:]); break
with open(f'{Wd}/notes.md','a') as f:
    f.write('\n- run 295: head edits DROPPED (edited = embed.numeric only, x5). VOCAB: z=(x-median)/iqr clipped +-10 (linear). NE arm key is "ne" (not nor).\n'
            '- run 296: local scan sel1 100 stays, 10 paths, (k2,k4) numeric cols 2,4: ' + '; '.join(f'{k}: {v}' for k,v in scan.items()) +
            f'\n  BEST (relerr vs NE truth 4.7/-2.35/-4.7/9.4/2.35) = {best} SAVED. local TF nll {tfres}\n')
```