`U_EMPTY` explains the constant offsets exactly:
- p_ins·U_EMPTY = −19.2, so the insulin unit is fully off on non-drug bundles.
- p_ne·U_EMPTY = −0.51, which gives the observed −1.02.

The writer's bias should have cancelled gelu(5 + p·U_EMPTY), not gelu(5). That part is easy to fix.

The harder problem is that my rebuilt u is not exact on drug bundles. Against the true projections from the tool, insulin agrees well at high rates (2.8 vs 2.6 at rate 2.7) but badly at low rates (0.39 true vs 1.02 rebuilt at rate 0.9). Norepinephrine agrees only loosely (corr 0.66). With a decoder this sensitive, any rebuild error becomes a dose error, and the x2 and x3 edits push rates outside the recorded range, where I can only rely on the rebuild.

So I need to find which part of the rebuild is wrong. I'll split the error (true minus rebuilt projection) by bundle composition:
- insulin rate;
- whether a norepinephrine start is in the same bundle;
- whether END atoms of the previous hour's infusions are present.

I'll also try the other `end_mode` options of `drug_atoms`. If one composition carries the error, that points to the bug. If the error depends only on rate, the bug is in how rate/dose are scored or binned.

```python
j0=ok_[hi[ok_]][0]
V_=drug_atoms(sidx[int(sid8[j0])], int(np.floor(hr8[j0])), 'both')
print('drug_atoms returns',type(V_), len(V_), type(V_[0]) if len(V_) else None, (np.asarray(V_[0]).shape if len(V_) else None))
for em_ in ['none','start','starts','end','ends','both','prev','all']:
    try:
        Vt=drug_atoms(sidx[int(sid8[j0])], int(np.floor(hr8[j0])), em_); print(em_,'n atoms',len(Vt))
    except Exception as ex: print(em_,'ERR',repr(ex)[:80])
# composition per bundle: previous-hour rates (END atoms) and NE in same bundle
I_=np.array([sidx[int(s)] for s in sid8[ok_]]); T_=np.floor(hr8[ok_]).astype(int)
prev_i=np.array([ins_all[i,t-1] if t>=1 else 0 for i,t in zip(I_,T_)]); prev_n=np.array([ne_all[i,t-1] if t>=1 else 0 for i,t in zip(I_,T_)])
cur_i=ri[ok_]; cur_n=rn[ok_]
res_i=imp_ins[ok_]-rec_i
mI=hi[ok_]&np.isfinite(res_i)
import pandas as pd
dfc=pd.DataFrame(dict(r=np.round(cur_i[mI]*2)/2, ne_now=(cur_n[mI]>0).astype(int), ins_prev=(prev_i[mI]>0).astype(int), ne_prev=(prev_n[mI]>0).astype(int), res=res_i[mI], imp=imp_ins[ok_][mI]))
print(dfc.groupby(['ne_now','ins_prev','ne_prev']).res.agg(['count','mean','std']).round(3))
print(dfc.groupby('r').agg(n=('res','count'),res=('res','mean'),imp=('imp','mean')).round(2).T)
```