My run-37 inspection code reused the name `g`, which overwrote the training hourly grid DataFrame. I will rebuild that grid when needed. This run lists the remaining tables, skipping `g`, so I can reuse the merged factual-output tables.

```python
for nm_ in ['gt','gtt2','gfm','mam','fa','m','de','r','gg']:
    v_ = globals()[nm_]; print(nm_, getattr(v_,'shape',None), list(v_.columns)[:45] if hasattr(v_,'columns') else type(v_))
print(gfm.head(3).to_string()); print(gt.head(3).to_string())
```