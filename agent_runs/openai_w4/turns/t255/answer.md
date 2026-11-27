The retained objects include both the factual replay and the separate insulin and NE response directions. The output limit again hid the impulse results, so I’ll inspect those directly. If the long-run response differs from the immediate calibration, we should fit a direction to the full response curve rather than merely increase its strength.

```python
print("=== RECORD KERNEL VARIABLES ===")
for name, obj in list(globals().items()):
    if "kernel" in name.lower() or re.search(r"137$", name):
        if isinstance(obj, dict):
            print(name, "keys", list(obj)[:6])
        elif isinstance(obj, np.ndarray):
            print(name, obj.shape, np.round(obj, 4).tolist() if obj.size <= 26 else "")
        elif isinstance(obj, pd.DataFrame):
            print(name, obj.shape, list(obj.columns))
        elif isinstance(obj, (float, int, str)):
            print(name, repr(obj)[:80])

print("\n=== ORIGINAL IMPULSE TABLES ===")
for drug, tab in impulse_tables_repair.items():
    print(drug)
    print(tab.to_string(index=False))

print("\n=== RETAINED EDIT IMPULSES ===")
for name in ["impulse_results159", "impulse_tests166"]:
    for key, val in globals()[name].items():
        print(name, key)
        if isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, dict):
                    print(" ", k, {kk: (getattr(vv, "shape", None)
                          if hasattr(vv, "shape") else repr(vv)[:100])
                          for kk, vv in list(v.items())[:6]})
                elif hasattr(v, "shape"):
                    print(" ", k, v.shape,
                          np.round(np.asarray(v), 3).tolist() if np.asarray(v).size < 20 else "")
                else:
                    print(" ", k, repr(v)[:450])

print("\n=== REPLAY IMPLEMENTATION ===")
import dis
dis.dis(replay_factual125)

print("\n=== RESPONSE-DIRECTION DIAGNOSTICS ===")
for name in ["preserved_b194", "ne_direction194", "ins_diff206", "ins_unit206"]:
    v = globals()[name]
    a = as_np184(v)
    print(name, a.shape, "norm", float(np.linalg.norm(a)))
for key, v in ne_directions206.items():
    print("NE ridge", key, "norm", float(np.linalg.norm(as_np184(v))))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 222\nInspecting retained impulse results and factual replay implementation, with compact output. "
            "Considering longitudinal response-direction calibration rather than further immediate-response scaling. "
            "No saved weights changed.\n")
```
