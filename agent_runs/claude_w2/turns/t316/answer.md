The local engine agrees with the tool, so I can now choose the scale factors locally and only spend forward passes on confirmation.

| Scale (dose, rate) | NE ×2 local / tool | NE ×½ local / tool | Insulin ×2 |
|---|---|---|---|
| (3,3) | 2.76 / 2.66 | −1.44 / −1.48 | −2.19 |
| (5,5) | 2.85 / 2.82 | −2.18 / −2.22 | −1.81 |
| (1,1) original | — | — | −2.18 (tool −2.14) |

- **Factual fit:** my local teacher-forced NLL improves at (5,5) (1.5234 → 1.5216), consistent with earlier tool results. (5,5) is saved; it is the same edit as before.
- **NE pattern:**
  - ×2 peaks near k=5 and then declines (2.37 at k=10).
  - ×½ keeps growing (−3.17 at k=10).
  - Stopping NE (×0) is under-predicted (−1.6 vs my estimate −4.7).
  - ×3 saturates near 4 (estimate 9.4).

  The model behaves as if very high or very low NE rates mark a different kind of patient. That is confounding baked into its representation, and plain scaling cannot remove it.
- **Dose vs rate column:**
  - Raising the dose column mostly strengthens the low-dose arms but hurts NE ×2/×3. It also weakens the main readout, insulin ×2: −1.81 at (5,5) and −1.66 at (10,5).
  - Raising the rate column helps NE ×½ and ×0 without hurting ×2, and slightly strengthens insulin. (5,10) gives insulin ×2 −1.94, ×½ +1.79, ×0 +2.05.

Next I scan a low dose factor with a high rate factor: (1,10), (2,10), (3,10), (2,15), (3,15), (1,15), with local factual NLL for each. I choose by summed relative error over all 10 arms: NE ×2, ×½, ×0, ×3, ×1.5, NE ×2 on cuff, insulin ×2, ×½, ×0 and insulin ×2 on lab. The candidate must keep local NLL no worse than the original, and the winner is saved.

- **Confirms:** some (low, high) setting keeps NE ×2 near 2.85 and ×½ near −2.3 while leaving insulin ×2 at or below −1.9.
- **Refutes:** NE ×2 and ×½ collapse whenever the dose factor is low. In that case (5,5) or (5,10) stays.

I also save 100 other tuning stays as a second check of the factual budget.

```python
import time, json, numpy as np, pandas as pd, traceback
Wd='/work'
def P(*a): print(*a, flush=True)
T0=time.time()
TA={'ne':{'n2':4.7,'n.5':-2.35,'n0':-4.7,'n3':9.4,'n1.5':2.35},'cuff':{'n2':4.7},'ins':{'i2':-16.5,'i.5':8.25,'i0':16.5},'lab':{'i2':-16.5}}
def score(v): return round(sum(abs(v[g][k]-t)/abs(t) for g,d in TA.items() for k,t in d.items()),3)
for k,v in scan.items(): v['score']=score(v)
tf_orig=1.5233968764190775
def tfn(k2,k4):
    r=teacher_forced(tun, sel1, getW(mk_loader(k2,k4))); return float(r['nll'].mean())
for kk in [(3,3),(5,5),(7,7),(5,10),(10,5),(10,10)]:
    try: scan[kk]['dtf']=round(tfn(*kk)-tf_orig,5)
    except Exception as ex: P('ERR tf',kk,ex)
P('prev', {k:(v['score'],v['relerr'],v.get('dtf')) for k,v in scan.items()})
for (k2,k4) in [(1,10),(2,10),(3,10),(2,15),(3,15),(1,15)]:
    if time.time()-T0>700: P('time guard'); break
    try:
        t1=time.time(); Wv=getW(mk_loader(k2,k4)); res=run_rollout(R,Wv,arms,n_paths=10,seed=0)
        v=dict(ins=agg(res,'insulin','glucose_finger',[1,2,3]), ne=agg(res,'norepinephrine','map_arterial',[4,5,6,7,8]),
               cuff=agg(res,'norepinephrine','map_cuff',[4]), lab=agg(res,'insulin','glucose_lab',[1]))
        v['relerr']=relerr(v); v['score']=score(v); v['dtf']=round(tfn(k2,k4)-tf_orig,5)
        scan[(k2,k4)]=v; P((k2,k4),'%.0fs'%(time.time()-t1),v)
    except Exception as ex: P('ERR',(k2,k4),traceback.format_exc()[-800:]); break
best=None
try:
    ok={k:v for k,v in scan.items() if k!=(1,1) and v.get('dtf',1)<=0.0}
    best=min(ok,key=lambda k: ok[k]['score']); P('BEST',best,ok[best])
    save_edit('embed.numeric.weight', mk_num(*best)); P('saved; edited', edited())
except Exception as ex: P('ERR save',traceback.format_exc()[-600:])
json.dump({str(k):v for k,v in scan.items()}, open(f'{Wd}/scan297.json','w'))
allt=sorted(int(x) for x in tun.stay_id.unique()); s1=set(sel1)
sel2=[s for s in allt if s not in s1][:100]; json.dump(sel2, open(f'{Wd}/sel2.json','w')); P('SEL2', json.dumps(sel2))
with open(f'{Wd}/notes.md','a') as f:
    f.write('\n- run 296 scan (local=tool within ~0.1): (1,1) ne2 1.19 ne.5 -.73; (3,3) 2.76/-1.44; (5,5) 2.85/-2.18 ne0 -1.6 ne3 3.98; (7,7) 2.68/-2.74; (10,10) 2.37/-3.17 -> NE x2 HUMP (confounding in repr). ins x2 ~-1.8..-2.2 for all k (no insulin lever).\n'
            '- run 297: score = sum rel err over NE(5 arms)+cuff+ins(3)+lab vs truth est; scan (k2 low,k4 high); results: ' + '; '.join(f'{k}: ne {v["ne"]} ins {v["ins"]} sc {v["score"]} dtf {v.get("dtf")}' for k,v in scan.items()) +
            f'\n  BEST saved = {best}. sel2.json = 100 other tuning stays for factual check.\n')
```