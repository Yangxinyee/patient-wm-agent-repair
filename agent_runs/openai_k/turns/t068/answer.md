The record-based estimates support an insulin effect near −2 per rate unit and a norepinephrine effect near +3; glucose effects persist longer. I’ll now verify the saved head derivatives and dose contrasts. If matching those effects requires large changes on factual inputs, I’ll reject a head-only correction.

```python
print("=== COUNTERFACTUAL TABLE ===")
for c in CF59:
    print({k: (tuple(v.shape) if hasattr(v, "shape") else v)
           for k, v in c.items()})
print("\nTilt vector:", beta59.tolist())
print("Codes:", repr(VOCAB["codes"])[:1200])
vc = VOCAB["value_bin_centres"]
print("Centre table:", type(vc).__name__,
      repr(vc)[:1800])

print("\n=== FEATURE HELPER REFERENCES ===")
for fun in (r59_features, r59_core):
    print(fun.__name__, "references:", fun.__code__.co_names)
    for name in fun.__code__.co_names:
        if name not in globals():
            continue
        value = globals()[name]
        if isinstance(value, (np.ndarray, torch.Tensor)):
            print(" ", name, tuple(value.shape),
                  value.tolist() if value.numel() <= 24 and isinstance(value, torch.Tensor)
                  else (value.tolist() if isinstance(value, np.ndarray) and value.size <= 24 else ""))
        elif isinstance(value, (list, tuple, dict)) and len(value) <= 12:
            print(" ", name, repr(value)[:500])

print("\n=== FIRST IDENTIFICATION NOTES ===")
print(open("/work/notes.md").read()[:4000])
print("\n=== FORWARD OUTPUT FILES ===")
import os
for folder in sorted(os.listdir("/work")):
    if folder.startswith("fwd_"):
        print(folder, os.listdir("/work/" + folder))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 62: verify correction statistics\n"
            "- Recovered DML effects: insulin/finger -1.935, insulin/lab -2.015; NE/arterial +3.053, NE/cuff +2.528.\n"
            "- Six-hour impulse estimates show longer glucose persistence than MAP persistence.\n"
            "- Inspecting counterfactual tables and exact head-response derivatives before fitting; no weights edited.\n"
            "- Python runs: 62; forward passes: 4.\n")
```