The last result shows no clear norepinephrine policy jump, and the transition reconstruction is close only in its summary state. I’ll now inspect the existing regression controls and dose-decoder construction. This will determine whether anchored contrasts can improve insulin sensitivity without adding harmful feedback; no weights will change in this step.

```python
print("=== AVAILABLE ANALYSIS FUNCTIONS ===")
fn_lines162 = []
for k, v in sorted(list(globals().items())):
    if isinstance(v, types.FunctionType) and not k.startswith("_"):
        fn_lines162.append(f"{k}{v.__code__.co_varnames[:v.__code__.co_argcount]}")
print("\n".join(fn_lines162)[:2300])

print("=== UNIT-TARGET CONSTRUCTION ===")
print(compact_dis160(head_unit_target)[:3100])
print("RANK TEMPLATE", np.asarray(rank_template))
print("GRAFT WEIGHT SHAPES",
      {k: np.asarray(v).shape for k, v in graft_original.items()})

print("=== TRANSITION SAMPLE ARRAYS ===")
sample_lines162 = []
for k, v in sorted(list(globals().items())):
    if k.startswith("rm_") and k != "rm_W":
        if isinstance(v, (np.ndarray, torch.Tensor)):
            desc = str(tuple(v.shape))
        elif isinstance(v, dict):
            desc = str(list(v)[:6])
        elif isinstance(v, types.FunctionType):
            desc = str(v.__code__.co_varnames[:v.__code__.co_argcount])
        else:
            desc = str(v)[:100] if isinstance(v, (int, float, str, list, tuple)) else type(v).__name__
        sample_lines162.append(k + " " + desc)
print("\n".join(sample_lines162)[:1900])
print("MATCHED COLUMNS", list(matched.columns))
print(matched[["stay_id", "hour", "event", "state_row", "dt_seconds"]].head(3).to_string(index=False))
```
