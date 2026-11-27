The last five replies were blocked, so I am resuming from the notebook. The notebook ends with these points:

- The v4 edit (a dedicated injection channel from insulin and norepinephrine rates into the state) had the wrong sign for insulin when tested in fwd_8.
- The edited rollouts drifted far from the original model's, by up to +26 mg/dL by hour 24, even in stays with no insulin.
- The likely cause is a leak: the channel also fires at measurement bundles, not only at drug bundles.

There are also files I have no notes on: `fwd_9` and `v5_parts.npz`. They probably come from the blocked runs. Python runs are now the scarce budget (24 left), so this run only takes stock:

- which edit is saved now
- what `fwd_9` holds
- what `v5_parts.npz` holds
- which helper functions and arrays still exist in the kernel

It prints only aggregate numbers and names, never raw records or stay ids, which may be what tripped the filter.

**What would confirm the current picture:** the edits match v4(1,1), `fwd_9` is a factual or dose run of v4, and the replica functions and arrays (`Tm`, `M`, `A`, `kv3`, `build_v4`) still exist. In that case, the next run removes the measurement-bundle leak by making the channel's readouts vanish on measurement tokens. If `v5_parts` turns out to be an existing leak fix, I test that directly instead.

```python
import os, json, numpy as np
out=[]
out.append('edited:'+str(edited()))
for n in edited():
    a=load(n); b=load_original(n); out.append(f'{n} |d|={np.linalg.norm(a-b):.4g} shape={a.shape}')
fl=sorted(os.listdir('/work/fwd_9')); out.append('fwd_9 files:'+str(fl))
for f in fl:
    p='/work/fwd_9/'+f
    if f.endswith('.json'):
        d=json.load(open(p))
        if isinstance(d,dict):
            out.append(f+': '+str({k:(v if not isinstance(v,(list,dict)) else type(v).__name__) for k,v in list(d.items())[:15]})[:500])
    elif f.endswith('.csv'):
        import pandas as pd
        df=pd.read_csv(p); out.append(f+f' rows={len(df)} cols={list(df.columns)[:12]}')
for fn in ['rep_drug','v4_parts','chan','v5_parts']:
    try:
        z=np.load('/work/'+fn+'.npz'); out.append(fn+': '+str({k:z[k].shape for k in z.files})[:500])
    except Exception as e: out.append(fn+' ERR '+str(e)[:80])
g=globals()
want=['Tm','qs','c','rbar','M','A','Wo','Wbin','tI','tN','gp','kv3','BB3','BB','build_v4','build_v5','prelude_b','sig_ref','aI3','aN3','aP3','qd','Xd','LN1','f_meas']
out.append('globals:'+str({n:getattr(g[n],'shape',type(g[n]).__name__) for n in want if n in g})[:700])
out.append('prelude tensors:'+str([n for n in NAMES if 'prelude' in n])[:700])
if 'v5_parts' in fl or os.path.exists('/work/v5_parts.npz'):
    z=np.load('/work/v5_parts.npz')
    for k in z.files:
        if z[k].size<10: out.append(f'v5 {k}={np.round(z[k],4).tolist()}')
with open('/work/notes.md','a') as f:
    f.write('\n## Restart 10 (run 277-)\n- Runs 272-276 outputs were blocked by a content filter: print only aggregates, never raw rows or stay ids.\n')
print('\n'.join(out))
```