fwd_8 compared the current edit with the original model on the 100 tuning stays:

| Readout | Current edit | Original |
|---|---|---|
| Insulin x2 | −2.70 | −2.01 |
| Insulin x1/2 | +0.79 | +1.16 |
| NE x2 | +0.02 | +0.37 |
| NE x1/2 | −0.00 | −0.06 |

The targets I estimated from the records are −12.5, +6.25, +4.33 and −2.17. So the edit is far too weak for insulin, and it removes the small norepinephrine effect the original model had. That is worse than doing nothing on that readout.

I have only 7 Python runs left, but 42 forward passes. So every Python run has to count. The last notes were written before fwd_7 and fwd_8. That means I don't know for sure which kv edit is in place (C1 or 0.5× of the full edit), or whether fwd_7's factual check passed.

Next is one compact diagnostic run. It will show:
- what the edited tensors are and whether they match C1;
- the fwd_7 factual NLL change, overall and early versus later hours;
- the fwd_8 shift for each hour;
- the tuning stay list I need for forward requests.

What would confirm my hypothesis: the hourly profile shows only a small, flat added insulin shift, and the NE added shift is negative. Then the NE channel would be counterproductive and should be removed, and insulin only needs a larger gain. What would refute it: the insulin injection appears in early hours and then decays. Then the problem is how the effect carries forward through the model's drawn readings, not the gain.

```python
import os, json, numpy as np, pandas as pd
def A(x):
    try: return x.detach().cpu().numpy()
    except AttributeError: return np.asarray(x)
kvn='core.prelude.0.attn.kv.weight'
try:
    for n in edited():
        d=A(load(n))-A(load_original(n)); print(n, d.shape, 'max %.4f fro %.4f'%(np.abs(d).max(), np.linalg.norm(d)))
    kvd=A(load(kvn))-A(load_original(kvn))
    for fn in ('C1.npz','edit_parts.npz'):
        Z=np.load('/work/'+fn); print(fn, {k:Z[k].shape for k in Z.files})
        for k in Z.files:
            if Z[k].shape==kvd.shape: print('  |kvd-%s|=%.4f |%s|=%.4f'%(k,np.linalg.norm(kvd-Z[k]),k,np.linalg.norm(Z[k])))
    DWf=np.load('/work/DW_kv_full.npy'); print('|kvd-0.5DWf|=%.4f |kvd|=%.4f'%(np.linalg.norm(kvd-0.5*DWf), np.linalg.norm(kvd)))
except Exception as ex: print('ERR1', repr(ex)[:200])
try:
    fn=[n for n,v in list(globals().items()) if callable(v) and not n.startswith('_') and getattr(v,'__module__','')=='__main__']
    print('funcs', fn[:80])
    print('scal', {n:v for n,v in globals().items() if n in ('kI','kN','gI','gN')})
except Exception as ex: print('ERR2', repr(ex)[:200])
for k in (7,8):
    d=f'/work/fwd_{k}'; print(k, sorted(os.listdir(d)))
def fact(k):
    d=f'/work/fwd_{k}'; R={}
    for m in ('original','edited'):
        f=f'{d}/factual_{m}.csv'
        if os.path.exists(f): R[m]=pd.read_csv(f)
    if not R: return
    df=next(iter(R.values())); print('fcols', list(df.columns))
    nc=[c for c in df.columns if 'nll' in c.lower()][0]
    ev=[c for c in df.columns if any(s in c.lower() for s in ('event','code'))]
    hc=[c for c in df.columns if 'hour' in c.lower()]
    for m,x in R.items():
        s=f'{m} nll={x[nc].mean():.4f}'
        if ev: s+=' '+str(x.groupby(ev[0])[nc].mean().round(4).to_dict())
        print(s)
    if len(R)==2 and len(R['original'])==len(R['edited']):
        dd=R['edited'][nc].values-R['original'][nc].values; print('dNLL %.4f'%dd.mean())
        if hc:
            h=R['original'][hc[0]].values; print('dNLL h<=5 %.4f  h>5 %.4f'%(dd[h<=5].mean(), dd[h>5].mean()))
try: fact(7)
except Exception as ex: print('ERR3', repr(ex)[:200])
try:
    for m in ('original','edited'):
        f=f'/work/fwd_8/dose_edit_{m}.csv'
        if not os.path.exists(f): continue
        df=pd.read_csv(f)
        if m=='original': print('dcols', list(df.columns), len(df)); print(df.iloc[0].to_dict())
        hc=[c for c in df.columns if 'hour' in c.lower()]; ec=[c for c in df.columns if 'edit' in c.lower()]
        sc=[c for c in df.columns if 'shift' in c.lower()]; rc=[c for c in df.columns if 'reach' in c.lower()]
        x=df[df[rc[0]].astype(bool)] if len(rc)==1 else df
        if ec and sc and hc:
            g=x.groupby([ec[0],hc[0]])[sc[0]].mean().unstack()
            for e,row in g.iterrows(): print(m[:4], e, np.round(row.values,1).tolist())
        elif hc:
            for s in sc: print(m[:4], s, np.round(x.groupby(hc[0])[s].mean().values,1).tolist())
except Exception as ex: print('ERR4', repr(ex)[:200])
try:
    S=json.load(open('/work/sel.json'))
    if isinstance(S,dict):
        print('sel keys', {k:(len(v) if hasattr(v,'__len__') else v) for k,v in S.items()}); s1=S.get('sel1', list(S.values())[0])
    else: s1=S
    print('sel1', json.dumps(s1))
except Exception as ex: print('ERR5', repr(ex)[:200])
open('/work/notes.md','a').write('\n## Session 9 (restart after 355 replies; 8 fwd, 293 runs)\n- fwd8 dose_edit current edit: ins x2 -2.70 (orig -2.01), ins half +0.79 (orig 1.16), ne x2 0.019 (orig 0.367), ne half -0.001. Insulin too weak; NE effect removed.\n- Replies were blocked by a content filter: keep outputs SHORT (stats only).\n')
```