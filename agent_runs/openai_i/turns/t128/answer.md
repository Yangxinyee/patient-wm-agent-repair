The terminal cases contain no infusion starts, so their target innovation can be set to zero. The saved record estimates show multi-hour effects, but they remain observational estimates—not intervention truth. I’ll recover the dose residuals and latent directions next. A correction must respond to dose changes without creating large shifts on ordinary factual bundles.

```python
print("=== POLICY RESIDUALS ===")
for _drug in ["insulin", "norepinephrine"]:
    _v = dose_residuals[_drug]
    print(_drug, type(_v).__name__,
          getattr(_v, "shape", None),
          str(list(_v.keys()))[:150] if isinstance(_v, dict) else "")
    if isinstance(_v, np.ndarray):
        print("finite", np.isfinite(_v).sum(),
              "quantiles", np.nanquantile(_v, [0, .01, .25, .5, .75, .99, 1]).round(4))
    print("dose model", type(dose_models[_drug]).__name__)
print("=== LATENT DIRECTION OBJECTS ===")
for _name, _obj in list(globals().items()):
    if _name.startswith("_") or not re.search("104|direction|decoder|110|111", _name):
        continue
    if isinstance(_obj, (np.ndarray, torch.Tensor)):
        print(_name, tuple(_obj.shape))
        if _obj.ndim == 1 and len(_obj) <= 64:
            print(" values", np.asarray(_obj).round(5).tolist())
    elif hasattr(_obj, "coef_"):
        print(_name, type(_obj).__name__, "coef", np.asarray(_obj.coef_).shape)
    elif isinstance(_obj, dict):
        print(_name, "dict", str(list(_obj.keys())[:10])[:180])
    elif inspect.isfunction(_obj):
        print(_name, "function", list(inspect.signature(_obj).parameters))
    elif isinstance(_obj, (float, int, str)):
        print(_name, repr(_obj)[:100])
    elif isinstance(_obj, (list, tuple)):
        print(_name, type(_obj).__name__, len(_obj))
print("=== IDENTIFICATION NOTES ===")
_a = _notes.find("## Identification checks")
_b = _notes.find("## ", _notes.find("## Lagged-effect investigation") + 3)
print(_notes[_a:_b][:3500])
print("=== VALIDATION AND LATENT NOTES ===")
for _heading in ["## Python 104", "## Python 110"]:
    _a = _notes.find(_heading)
    _b = _notes.find("\n## ", _a + 3)
    print(_notes[_a:_b if _b >= 0 else None][:1200])
print("drug posterior RMS",
      float(torch.sqrt(torch.mean((drug_z110 - Ztarget84[drug_idx110])**2))))
print("hidden perturbation shapes",
      {str(k): np.asarray(v).shape for k, v in hidden_edits110.items()})
```
