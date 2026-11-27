All 53 unmatched bundles occur at hour 25.0833, have zero starting rates, and lie beyond the panel’s prediction window. They appear to be terminal END-only bundles, not missing treatment observations. I’ll verify their contents and recover the saved innovation estimates and latent directions. Nonzero starts among these rows would refute that explanation.

```python
import torch, dis

print("=== TERMINAL BUNDLE CHECK ===")
print("unmatched all zero:", bool(np.all(drug_rates110[_bad] == 0)))
print("unmatched hours:", np.unique(state_hours[drug_idx110[_bad]]))
print("case element types:",
      [(type(x).__name__, getattr(x, "shape", None),
        len(x) if isinstance(x, (list, dict, tuple)) else None)
       for x in drug_cases110[_bad[0]]])
print("=== SMALL DICTIONARIES ===")
for _name, _obj in list(globals().items()):
    if _name.startswith("_") or not isinstance(_obj, dict):
        continue
    if len(_obj) <= 40 and _name not in ("__builtins__",):
        print(_name, len(_obj), str(list(_obj.keys())[:8])[:180])
print("=== FUNCTION CONTRACTS ===")
for _name, _obj in list(globals().items()):
    if not _name.startswith("_") and inspect.isfunction(_obj) and _obj.__module__ == "__main__":
        if re.search("repair|latent|post|core|step|action|pool|head|mlp|atom", _name):
            print(_name, list(inspect.signature(_obj).parameters))
for _name in ["posterior109", "pool109", "repair_linear", "repair_mlp"]:
    if _name in globals() and inspect.isfunction(globals()[_name]):
        _fn = globals()[_name]
        print("BYTECODE SUMMARY", _name,
              "globals", _fn.__code__.co_names,
              "constants", [x for x in _fn.__code__.co_consts
                            if isinstance(x, (str, int, float, tuple))][:15])
print("=== EFFECT ESTIMATES ===")
print(irf_table.to_string(index=False))
print("IRF coefficients:", {k: np.asarray(v).round(3).tolist()
                            for k, v in irf_coefficients.items()})
print("=== IDENTIFICATION NOTES ===")
_a = _notes.find("## Identification checks")
_b = _notes.find("## ", _notes.find("## Lagged-effect investigation") + 3)
print(_notes[_a:_b][:3000])
print("=== PERTURBATION SHAPES ===")
print({str(k): np.asarray(v).shape for k, v in hidden_edits110.items()})
print("Posterior reconstruction RMS:",
      float(torch.sqrt(torch.mean((drug_z110 - Ztarget84[drug_idx110])**2))))
```
