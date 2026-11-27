The notebook contains matched rollout targets, so the immediate drug coefficients need not be extrapolated into sustained effects. I’ll retrieve those targets and the exact paired-neuron construction. This will show whether the current edit is actually closer to the records and whether norepinephrine gating is a useful next intervention.

```python
print("EXACT MATCHED TARGETS")
print(target_results256)
print("\nTWELVE-LAG RECORD KERNELS")
for drug, values in kernels137.items():
    print(drug, values)
print("\nMATCHED-TARGET CONSTRUCTION NOTE")
for s in sections281:
    if re.match(r"## Run (238|240|256)\b", s):
        print(s[:1000])

print("\nPAIR CONSTRUCTION")
for fn in (symmetric_ins265, symmetric_ne271):
    print(fn.__name__, "globals", fn.__code__.co_names)
    print("constants", [c for c in fn.__code__.co_consts if isinstance(c, (str, int, float, tuple))])
    print(" | ".join(f"{i.opname} {i.argrepr}" for i in dis.get_instructions(fn))[:3000])

print("\nACTION FIXTURE")
print(action_fixture_repair.columns.tolist())
print(action_fixture_repair.head(3).to_dict("records"))
print("ATOM ENCODER", atom_clean58.__code__.co_names)
print("NE INNOVATION COEFFICIENT BLOCK NORMS",
      [float(np.linalg.norm(coef_ne186[a:b])) for a,b in ((0,192),(192,384),(384,576),(576,577))])
for k in set(symmetric_ne271.__code__.co_names + symmetric_ins265.__code__.co_names):
    if k not in globals():
        continue
    obj = globals()[k]
    if hasattr(obj, "shape"):
        print(k, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, (int, float)):
        print(k, obj)
    elif isinstance(obj, dict):
        print(k, "dict", len(obj), list(obj)[:5])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 284\nRecovered exact matched record-kernel targets and paired-neuron construction to select a specific calibration. No weights changed. Targets remain estimates under the longitudinal identification assumptions, not supplied simulator truth.\n")
```
