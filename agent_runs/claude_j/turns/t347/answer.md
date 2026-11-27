Run 293 failed before it did anything: `fb_` got the string `'value'` where it needed a number. So that run taught me nothing.

I have 7 Python runs and 39 forward passes left, so I will re-orient as cheaply as I can. My notes stop at "Test mk2(12,12)", but folders `fwd_10` and `fwd_11` exist and are not written up. They are probably the tool's results for that targeted edit, which amplifies head-2 value only along the dose directions. This one run will:
1. Check what is saved in `kv.weight`. A rank-2 change means mk2; a full-rank change means uniform scaling.
2. Compare file times, to see whether fwd_10/11 ran after that edit was saved.
3. Summarise fwd_8 to fwd_11 in a few short lines.
4. Recover the stay ids and the split I need for further forward requests.

What would confirm my belief: a rank-2 edit saved before fwd_10/11, with fwd_10/11 holding a dose_edit and a factual result. If the folders hold something else, I will spend two forward passes testing the current edit directly.

```python
import json, os, glob, numpy as np, pandas as pd
nm='core.prelude.0.attn.kv.weight'
try:
    W0=np.asarray(load_original(nm),dtype=np.float64); W1=np.asarray(load(nm),dtype=np.float64)
    D=W1-W0; rn=np.linalg.norm(D,axis=1); nz=np.where(rn>1e-7)[0]
    print('kv',W0.shape,'rows',(int(nz.min()),int(nz.max()),len(nz)) if len(nz) else None,'|D| %.4g |V2| %.4g'%(np.linalg.norm(D),np.linalg.norm(W0[288:336])),'sv',np.round(np.linalg.svd(D[288:336],compute_uv=False)[:4],3))
except Exception as e: print('E1',repr(e)[:150])
try:
    c=mk2.__code__; print('mk2',c.co_varnames[:c.co_argcount],mk2.__defaults__,c.co_names)
except Exception as e: print('E2',repr(e)[:150])
try:
    ps={'f%d'%k:'/work/fwd_%d'%k for k in range(7,12)}
    for f in os.listdir('/work/edited'): ps['ed:'+f]='/work/edited/'+f
    ps['notes']='/work/notes.md'; m={k:os.path.getmtime(v) for k,v in ps.items()}; t0=min(m.values())
    print('mtime min:',{k:round((v-t0)/60,1) for k,v in m.items()})
except Exception as e: print('E3',repr(e)[:150])
for k in range(8,12):
    try:
        for f in sorted(os.listdir('/work/fwd_%d'%k)):
            if not f.endswith('.csv'): print(k,f); continue
            df=pd.read_csv('/work/fwd_%d/%s'%(k,f)); cols=list(df.columns)
            sc=[c for c in cols if 'stay' in c.lower()]; ns=df[sc[0]].nunique() if sc else -1
            nl=[c for c in cols if 'nll' in c.lower()]; ev=[c for c in cols if c.lower() in ('event','code')]
            if f.startswith('factual') and nl:
                s='nll %.5f '%df[nl[0]].mean()+' '.join('%s %.4f'%(a,b) for a,b in df.groupby(ev[0])[nl[0]].mean().items()) if ev else 'nll %.5f'%df[nl[0]].mean()
            else:
                ob=[c for c in cols if df[c].dtype==object]; num=[c for c in df.select_dtypes('number').columns if c not in sc and c.lower() not in ('hour','minute')]
                s=str(df.groupby(ob[0])[num].mean().round(3).to_dict()) if ob else str(df[num].mean().round(3).to_dict())
            print(k,f,len(df),ns,cols if k==11 else '',s[:300])
    except Exception as e: print('E4',k,repr(e)[:150])
try:
    f=sorted(glob.glob('/work/fwd_11/*.csv'))[0]; df=pd.read_csv(f); sc=[c for c in df.columns if 'stay' in c.lower()][0]
    sid=sorted(map(int,df[sc].unique())); f8=sorted(glob.glob('/work/fwd_8/*.csv'))[0]; s8=sorted(map(int,pd.read_csv(f8)[sc].unique()))
    tu=pd.read_csv('/data/tuning.csv'); tc=[c for c in tu.columns if 'stay' in c.lower()][0]
    print('n',len(sid),'same as fwd8',sid==s8,'in tuning %.2f'%np.isin(sid,tu[tc].unique()).mean()); print(','.join(map(str,sid)))
except Exception as e: print('E5',repr(e)[:150])
with open('/work/notes.md','a') as fh: fh.write('\n## Session 12 (run 294+): 7 py runs left. Run294: inspect current kv edit, summarize fwd_8..11 (10/11 likely mk2(12,12) tests).\n')
```