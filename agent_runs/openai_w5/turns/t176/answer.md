The rollout files include hourly shifts and uncertainty, which will help locate the weak halving response. I’ll recover the exact gate and exposure formulas and summarize shifts by hour. Consistent asymmetry from the first affected hour would implicate the decoder; asymmetry emerging later would point more toward feedback.

```python
def describe152(k, v):
    if isinstance(v, (np.ndarray, torch.Tensor, pd.DataFrame, pd.Series)):
        return f"{k}: {type(v).__name__}{tuple(v.shape)}"
    if isinstance(v, dict):
        return f"{k}: dict keys={list(v)}"
    if isinstance(v, (list, tuple)):
        return f"{k}: {type(v).__name__}({len(v)})"
    if inspect.isfunction(v):
        return f"{k}: function{inspect.signature(v)}"
    return f"{k}: {type(v).__name__} {str(v)[:160]}"

objects152 = [(k, v) for k, v in snapshot150
              if re.search(r"_r14[789]_", k)]
print("=== RUNS 147–149 OBJECTS ===")
print("\n".join(describe152(k, v) for k, v in objects152)[:6900])

print("=== HOURLY REACHED-ROW SHIFTS ===")
hour152 = dose151[dose151.reached.astype(bool)].groupby(["edit", "hour"]).agg(
    shift=("shift", "mean"), se_paths=("shift_se", "mean"), n=("stay_id", "size")
)
print(hour152.round(3).to_string())

print("=== VALUE HEAD SHAPES ===")
for name152 in NAMES:
    if name152.startswith("attr_heads.value_bin."):
        print(name152, SHAPES[name152])
```
