cand3 at full strength bought very little:

| Edit | cand3 | Original | Numerics ×3 alone | My estimate of truth |
|---|---|---|---|---|
| Insulin ×2 | −3.68 | −2.14 | — | about −16 |
| Insulin ×½ | +3.34 | — | — | — |
| NE ×2 | +2.82 | +1.08 | +2.66 | +4.7 |
| NE ×½ | −2.22 | — | −1.48 | −2.4 |

The head readout costs +0.044 nats of factual fit, about 9× the budget. To fit the budget it would need scaling to about 0.3, which would leave a negligible gain of roughly −0.4 on insulin ×2. So I am dropping the head readout entirely.

The NE effect comes from the dose-numeric amplification:
- **NE ×½:** at ×5 it is −2.22, close to my estimate of −2.4.
- **NE ×2:** it saturates (2.66 at ×3, 2.82 at ×5), probably because large positive dose deviations are squashed inside the drug encoder.

This run therefore does three things:
1. Revert the head and embed_transition edits, keeping numerics ×5.
2. Check how the rate numerics are normalised (linear, log, clipped), which would explain the ×2 saturation.
3. Use my validated local rollout engine to scan numeric scale factors on 30 of the stays, including separate factors for the dose and rate columns, and also the ×0, ×1.5 and ×3 NE arms.

The (1,1) row should reproduce my earlier validation numbers. If some setting raises NE ×2 toward 4.7 without overshooting ×½, I will adopt it. If every setting saturates around 3, I will keep about ×5.

```python
import os, re, json, time, traceback
import numpy as np, pandas as pd, torch
Wd='/work'
def P(*a): print(*a, flush=True)
for n in ['attr_heads.value_bin.0.bias','attr_heads.value_bin.0.weight','attr_heads.value_bin.2.bias','attr_heads.value_bin.2.weight','embed_transition.value_bin.weight']:
    try: drop_edit(n)
    except Exception as ex: P('ERR drop',n,ex)
P('edited now', edited())
try:
    P('VOCAB keys', list(VOCAB.keys()))
    for key in VOCAB:
        if 'norm' in key.lower() or 'numeric' in key.lower():
            P(key, json.dumps(VOCAB[key])[:1000])
except Exception as ex: P('ERR vocab',ex)
try:
    arch=open('/model/ARCHITECTURE.md').read().splitlines()
    hits=[(i,l) for i,l in enumerate(arch) if re.search(r'numeric|normalis|clip|z_rate|z_dose', l, re.I)]
    for i,l in hits[:20]: P('A',i,l[:220])
except Exception as ex: P('ERR arch',ex)
src=open(f'{Wd}/rollout.py').read()
def func_src(src,name):
    m=re.search(r'^def '+name+r'\(.*?(?=^def |\Z)', src, re.S|re.M)
    return m.group(0) if m else ''
for nm,cap in [('arm_embed',1500),('readout',1500)]:
    P('-----',nm); P(func_src(src,nm)[:cap])
keys=set(re.findall(r"arm\.get\(\s*['\"](\w+)['\"]", src))|set(re.findall(r"arm\[\s*['\"](\w+)['\"]\s*\]", src))
P('arm keys', keys)
def fmt(o):
    try:
        if isinstance(o,dict): return {k:(round(float(v),3) if np.isscalar(v) else str(v)[:60]) for k,v in o.items()}
        a=np.asarray(o,dtype=float); return np.round(a.ravel()[:12],3).tolist()
    except Exception: return str(o)[:300]
def ro(R,res,arms,drugs,outcome):
    last=None
    for d in drugs:
        try: return d, fmt(readout(R,res,arms,drug=d,outcome=outcome)[2])
        except Exception as ex: last=ex
    return 'ERR', str(last)[:200]
scan={}
try:
    NEK=next((k for k in keys if 'ins' not in k and not k.endswith('_from')), 'nor')
    tun=pd.read_csv('/data/tuning.csv'); sel1=json.load(open(f'{Wd}/sel1.json')); ids=sel1[:30]
    t1=time.time(); R=prep_rollout(tun, ids); P('prep %.1fs'%(time.time()-t1), 'NEK',NEK)
    is_t=isinstance(load('embed.numeric.weight'),torch.Tensor)
    b=np.asarray(load_original('embed.numeric.weight'),dtype=np.float32)
    def mk_loader(k2,k4):
        a=b.copy(); a[:,2]*=k2; a[:,4]*=k4
        at=torch.from_numpy(a) if is_t else a
        return lambda n: at if n=='embed.numeric.weight' else load(n)
    arms=[{}, {'ins':2.0}, {'ins':0.5}, {NEK:2.0}, {NEK:0.5}, {NEK:0.0}, {NEK:3.0}, {NEK:1.5}]
    for (k2,k4) in [(1,1),(5,5),(8,8),(5,10),(10,5),(2,8)]:
        t1=time.time()
        Wv=getW(mk_loader(k2,k4)); res=run_rollout(R,Wv,arms,n_paths=20,seed=0)
        oi=ro(R,res,arms,['insulin'],'glucose_finger'); on=ro(R,res,arms,['norepinephrine','nor','ne'],'map_arterial')
        scan[(k2,k4)]=(oi,on); P((k2,k4),'%.0fs'%(time.time()-t1),'INS',oi,'NE',on)
except Exception as ex:
    P('ERR scan', traceback.format_exc()[-1500:])
with open(f'{Wd}/notes.md','a') as f:
    f.write('\n- fwd_9 = sel1: cand3 factual 1.57204 vs orig 1.52793 (+0.044; gf 1.559->1.659, gl 1.227->1.324, MAP ok). fwd_10 cand3 dose_edit sel1: ins x2 -3.68, x.5 +3.34, NE x2 2.82, x.5 -2.22.\n'
            '- DECISION run 295: head readout dropped (too costly, too weak). Kept numeric cols 2,4 x5. Local scan of numeric (k2,k4) arms [fact, ins2, ins.5, ne2, ne.5, ne0, ne3, ne1.5] on sel1[:30]:\n'
            + '\n'.join(f'  {k}: {v}' for k,v in scan.items()) + '\n')
```