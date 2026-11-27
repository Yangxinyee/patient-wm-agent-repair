Arrays aligned with `Xf` exist for code, last bin, observed bin, the original logits and NLL (`L0`, `LS0`, `nll0`), and lag-1 dose (`dI1`, `dN1`). The ridge-decoder dictionary `DEC` also exists. Next I'll check what `DEC` holds and print short summary stats for these aligned arrays, so I can fit a decoder for the head-level edit and measure its factual cost.

```python
print({k_: (getattr(v_,'shape',None) if not isinstance(v_,dict) else list(v_.keys())[:6]) for k_, v_ in DEC.items()} if isinstance(DEC, dict) else type(DEC))
for nm in ['hf','pf','uf','sf','isg','ism','pg','pg2','obf','sT','mt','dI1','dN1','d1_','s_','t_']:
    a_ = g[nm]
    try: print(nm, a_.dtype, np.round([np.nanmin(a_), np.nanmean(a_), np.nanmax(a_)],3), "nuniq", len(np.unique(a_)))
    except Exception as e: print(nm, a_.dtype, str(e)[:40])
print("nll0 by code", [np.round(nll0[cf==c].mean(),4) for c in range(4)], "n", np.bincount(cf))
```