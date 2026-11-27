The recovered records support a negative insulin effect, with the sustained doubling target near −11 glucose units. The norepinephrine estimates are still observational and less secure. I’ll inspect the graft algebra and separator coefficients next so I can test finger-only gating without introducing an unintended lab or MAP change.

```python
import inspect, dis, numpy as np, torch

print("GRAFT AND EVALUATION FUNCTIONS")
for fname in ["make_contrast_graft", "sc_evaluate", "sc_cache_inputs"]:
    fn = globals()[fname]
    print("\n", fname, inspect.signature(fn))
    try:
        print(inspect.getsource(fn))
    except Exception:
        print("CONSTANTS", fn.__code__.co_consts)
        print("BYTECODE", "\n".join(
            f"{i.opname} {i.argrepr}" for i in dis.Bytecode(fn)
        ))

print("\nESSENTIAL GRAFT PARAMETERS")
for name in ["sc_nodes", "sc_rank", "sc_bias_comp", "sc_gate_b",
             "sc_gate_w", "r3_C", "r3_Q"]:
    obj = globals()[name]
    if isinstance(obj, np.ndarray):
        print(name, obj.shape, "norm", float(np.linalg.norm(obj)),
              "values" if obj.size <= 20 else "range",
              obj if obj.size <= 20 else [float(obj.min()), float(obj.max())])
    else:
        print(name, obj)
print("r3 candidate 0.1,1 structure",
      {k: (tuple(v.shape) if hasattr(v, "shape") else repr(v)[:150])
       for k, v in r3_candidates[(0.1, 1.0)].items()})

print("\nSEPARATOR-RELATED OBJECTS")
for name in sorted(list(globals())):
    if name.startswith("_") or not any(s in name.lower() for s in ["sep", "finger", "lab"]):
        continue
    obj = globals()[name]
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(name, "keys", list(obj.keys())[:12])
    elif not callable(obj):
        print(name, repr(obj)[:140])

print("\nLAST GRAFT INSTALLATION NOTES")
for section in sections:
    if any(s in section.splitlines()[0] for s in [
        "Protected synthetic-contrast", "Reversible protected-contrast",
        "Contrast-strength trial"
    ]):
        print(section[:1400])
```
