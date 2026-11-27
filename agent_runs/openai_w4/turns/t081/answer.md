The kernel still has the state fixtures, head features and dose-innovation analyses. The listing was again too large, so I’ll now request only the causal summaries and key function definitions. This should establish whether the records support a persistent drug response and identify where a factual-preserving correction could enter.

```python
print("DICT NAMES:", [n for n, v in list(globals().items())
                     if not n.startswith("_") and isinstance(v, dict)])
print("\nIMMEDIATE EFFECTS")
print(dml_results.to_string(index=False))
print("\nLAG MOMENTS")
print(tab_imp55.to_string(index=False))
print("\nBASELINE ROLLOUT")
print(base_roll.groupby("edit")["shift"].agg(["mean", "count"]).to_string())
print("\nSTATE BANKS")
for n, v in list(globals().items()):
    if isinstance(v, dict) and "x" in v and "stay_id" in v:
        print(n, {k: tuple(np.shape(v[k])) for k in v})
print("\nHEAD WEIGHTS")
for n in NAMES:
    if "attr_heads.value_bin" in n:
        print(n, SHAPES[n])
print("\nFUNCTION SOURCES")
for n in ("head_features_fixed", "memory67", "self_block67"):
    try:
        print(n + ":\n" + inspect.getsource(globals()[n]))
    except Exception as e:
        print(n, type(e).__name__, str(e)[:100])
```
