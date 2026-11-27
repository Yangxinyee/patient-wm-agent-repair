The tensor list confirms the value head is a two-layer MLP shared by all measurement codes, so any readout needs code-specific gating. Next I'll check what head-side helpers already exist in the kernel (signatures only) and what columns the saved factual output has, so I can validate a local value-head computation against the tool's predicted means.

```python
for fn in ['pred_test','check','check2','HN','ctx_nf','G_','lev','fb','zs','zb','Wd','rm','err_var','zcheck3','zpair','anchor_fit']:
    f=globals().get(fn)
    try: print(fn, inspect.signature(f) if callable(f) else type(f))
    except Exception as e: print(fn,'?')
fa=pd.read_csv('/work/fwd_2/factual_original.csv'); print(fa.shape, list(fa.columns))
```