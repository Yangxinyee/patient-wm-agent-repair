The retained objects include the causal estimates, exact atom recipes, exported states, and a usable value-head reconstruction. No weights have changed. I’ll now inspect the remaining bundle-building objects and the problematic norepinephrine records. A consistent omitted event or attribute would support a local encoding fix; inconsistent errors would argue against it.

```python
print("=== REMAINING INVENTORY ===")
print(inventory61[8800:14600])
print("\n=== ENCODING RECIPES ===")
for key in ["atom_recipe_repair", "measurement_recipe_repair", "head_cfg_repair"]:
    print(key, globals().get(key))
print("\n=== RUN-60 OBJECTS ===")
for key, val in sorted(list(globals().items())):
    if key.endswith(("58", "59", "60")):
        if isinstance(val, (np.ndarray, pd.DataFrame)) or torch.is_tensor(val):
            print(key, type(val).__name__, tuple(val.shape))
        elif callable(val):
            print(key, "function")
print("\n=== RETAINED SOURCE STRINGS ===")
for key, val in sorted(list(globals().items())):
    if isinstance(val, str) and len(val) > 150 and key not in {"inventory61"}:
        print(key, len(val), repr(val[:100]))
print("\n=== PROBLEM RECORDS ===")
problem_records63 = pd.read_csv(
    "/data/train.csv", usecols=["stay_id", "hour", "event", "value", "rate", "rate_unit", "dose", "dose_unit", "end_hour"]
)
print(problem_records63.loc[
    problem_records63.stay_id.isin([1010000074, 1010000094]) &
    problem_records63.hour.between(-1, 9)
].to_string(index=False))
```
