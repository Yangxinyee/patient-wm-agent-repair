The installation logic uses a sensor gate to keep the MAP correction out of glucose predictions, which gives us a way to change norepinephrine sensitivity without directly altering the insulin edit. I’ll now compare the cached factual and doubled-dose inputs. If they contain a direction that separates doubled doses from ordinary records, we can test it; otherwise further amplification is likely to be costly.

```python
print("=== CACHED STOCHASTIC INPUTS ===")
for k, v in list(globals().items()):
    if re.search(r"(mc|contrast|nebase|cal190|cal191)", k, re.I):
        if isinstance(v, dict):
            print(k, "dict", [(str(a), getattr(b, "shape", type(b).__name__))
                             for a, b in list(v.items())[:8]])
        elif isinstance(v, (np.ndarray, pd.DataFrame, torch.Tensor)):
            print(k, type(v).__name__, tuple(v.shape))
        elif callable(v):
            print(k, "function", v.__code__.co_varnames[:v.__code__.co_argcount])
        else:
            print(k, type(v).__name__, str(v)[:100])

print("\n=== EVALUATOR CONSTANTS ===")
for name in ["head177_eval", "wm_rawloss152", "map172_tau", "scalar_mean192"]:
    print(name, globals()[name].__code__.co_consts)
print("head177_names", head177_names)
print("wm_v151", type(wm_v151).__name__, getattr(wm_v151, "shape", None), wm_v151)
print("\n=== MAP FRONTIER ===")
print(map192_frontier.to_string(index=False)[:2800]
      if "map192_frontier" in globals() else "No frontier variable.")
print("\n=== META ===")
for name in ["wm_train145", "wm_tune145", "wm_meta146", "wm_meta147"]:
    if name in globals():
        obj = globals()[name]
        print(name, type(obj).__name__, getattr(obj, "shape", None),
              list(obj.columns) if isinstance(obj, pd.DataFrame) else str(obj)[:150])
```
