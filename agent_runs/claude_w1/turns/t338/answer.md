The helpers have no stored source, but the signatures are informative. `q_of(prev, dt, eps=None)` evidently takes the z-prior noise as an argument. So the replica can compute the head query under prior draws, which is exactly what is needed to test whether z-draw noise explains the v4 failure. A test could then show:
- the big decoder units' pre-activations swing by tens of units across z draws: hypothesis confirmed, and the redesign must put z-draw variance into the decoder's noise model;
- they barely move: the mismatch is elsewhere and head-level decoding is too unreliable to pursue.

I need the signatures of the row-level helpers to build that test, so I'll print them along with the structure the per-stay runner returns.

```python
for f in ['q_rows','run_rows','run_rows_wc','rows_of','run_stay','wc_vec','zs_','zsc','zb','vfun','run_head','solve_unit','true_shift','truth_rows','proc_stay','head_value2','eval3','fit_dec']:
    if f in globals(): print(f, str(inspect.signature(globals()[f]))[:150])
s0=int(ids[0])
try:
    r=run_stay(s0) if 'run_stay' in globals() else None
    print('run_stay type', type(r), (len(r) if hasattr(r,'__len__') else ''))
    if isinstance(r,(list,tuple)) and len(r): print(' elem', type(r[0]), (r[0].keys() if isinstance(r[0],dict) else (len(r[0]) if hasattr(r[0],'__len__') else r[0])))
    if isinstance(r,dict): print(' keys', list(r.keys())[:20])
except Exception as e: print('run_stay err', repr(e)[:300])
```