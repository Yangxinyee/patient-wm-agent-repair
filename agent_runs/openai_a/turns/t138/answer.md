The transition uses pre-normalized attention with a memory of earlier bundles. Matching the posterior alone is therefore insufficient. I’ll recover the validated measurement settings and exact state layout next, then test a complete local update. Small errors across both slots and output would support using it for dose contrasts.

```python
print("=== VALIDATED INSULIN ENCODING ===")
print("spec:", r96_best["spec"])
print("best objects:", {k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
                        for k, v in r96_best.items()})
print("state layout:", {k: (v.shape, str(v.dtype)) for k, v in rm_states.items()})
print("pure insulin indices:", rm_pure_idx[:8])
print("first insulin row:", rm_rows[0])

print("\n=== RECENT MEASUREMENT RECONSTRUCTION VARIABLES ===")
for nm, ob in list(globals().items()):
    if not nm.startswith(("r11", "_r11")):
        continue
    if isinstance(ob, (np.ndarray, torch.Tensor, pd.DataFrame)):
        print(nm, tuple(ob.shape))
    elif isinstance(ob, dict):
        print(nm, "keys", str(list(ob.keys())[:10])[:180])
    elif isinstance(ob, (str, int, float, bool, tuple)):
        print(nm, str(ob)[:180])
    elif isinstance(ob, list):
        print(nm, "length", len(ob), "first", str(ob[:1])[:180])

print("\n=== VOCABULARY ===")
print("keys:", list(VOCAB))
print("codes:", VOCAB["codes"])
print("config:", VOCAB["config"])
for nm in ("core.slots", "core.flow.a", "core.delta_in", "core.a_in"):
    ar = as_numpy(load_original(nm))
    print(nm, ar.shape, "range", float(ar.min()), float(ar.max()))

print("\n=== NORMALIZATION ENTRIES ===")
for key in (("glucose_finger", "value"), ("glucose_lab", "value"),
            ("map_arterial", "value"), ("map_cuff", "value"),
            ("insulin", "rate"), ("insulin", "dose"),
            ("norepinephrine", "rate"), ("norepinephrine", "dose")):
    print(key, rm_normspec.get(key))

print("\n=== FIRST IDENTIFICATION NOTES ===")
print(Path("/work/notes.md").read_text()[:3700])
```