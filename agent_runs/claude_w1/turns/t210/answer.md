The source code is not stored, so I will reconstruct the functions from their signatures, the globals they reference and their constants. I will also look at the small dicts (EDR, DEC, keysI, keysN). That should be enough to see whether vI and vN are finished decoders that the edit can use directly.

```python
for fn in ['proc_stay','cell_resid','solve_unit','diag','get_w','dose_arr','run_rows_wc','head_mean' if 'head_mean' in globals() else 'run_head']:
    f_=globals()[fn]; c=f_.__code__
    print('==',fn,str(inspect.signature(f_))[:120],'| glob:',[n for n in c.co_names][:25],'| consts:',[k for k in c.co_consts if isinstance(k,(int,float,str)) and not (isinstance(k,str) and len(k)>30)][:15])
print('EDR',{k:(type(v).__name__, getattr(v,'shape',None) if not isinstance(v,(int,float)) else v) for k,v in EDR.items()})
print('DEC',list(DEC.keys())[:10]); print('keysI',keysI,'keysN',keysN)
print('vI norm',np.linalg.norm(vI),'vN norm',np.linalg.norm(vN), 'w_ norm',np.linalg.norm(w_), 'co_',co_.shape)
```