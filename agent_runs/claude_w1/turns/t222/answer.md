The 5177 setA rows have replica probabilities (P64, Pm), observed bins (obin) and head inputs (wc_), but no per-row code array. The larger train table (Xf, 70,738 rows) is better for estimating factual cost, if I can attach observed values. I will check what GEN (per-stay output of proc_stay) and prow hold.

```python
g0=GEN[0]; print(type(g0).__name__, (list(g0.keys()) if isinstance(g0,dict) else len(g0)))
if isinstance(g0,dict):
    for k,v in g0.items(): print(' ',k, getattr(v,'shape',None) or (len(v) if hasattr(v,'__len__') else v))
else:
    for j,v in enumerate(g0[:12]): print(' ',j,type(v).__name__, getattr(v,'shape',None) or (len(v) if hasattr(v,'__len__') else v))
print('prow0',str(prow[0])[:200]); print('offs',offs[:5], 'sf',sf[:5],'hf',hf[:5],'pf',pf[:5],'uf',uf[:5],'lbf',lbf[:5])
```