The kernel still has the causal analyses and reconstruction helpers, but the inventory was too long to reveal the newest objects. No weights are edited. I’ll narrow the inspection to the stochastic caches and response estimates. Consistent estimates across identification methods would support a dose-response target; disagreement would argue for a more conservative correction.

```python
print("=== NEWEST STOCHASTIC OBJECTS ===")
for name, obj in list(globals().items()):
    if any(s in name for s in ["145", "146", "mc_groups", "stoch"]):
        if isinstance(obj, dict):
            print(name, "dict", list(obj.keys())[:20])
            for k, v in list(obj.items())[:12]:
                print(" ", k, tuple(v.shape) if hasattr(v, "shape") else
                      list(v.keys())[:8] if isinstance(v, dict) else type(v).__name__)
        elif hasattr(obj, "shape"):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif callable(obj):
            print(name, "function")
        else:
            print(name, str(obj)[:200])

print("\n=== CONTRAST DICTIONARIES ===")
for name, obj in list(globals().items()):
    if isinstance(obj, dict) and ("insulin", 2.0) in obj:
        print(name, list(obj.keys()))
        g = obj[("insulin", 2.0)]
        if isinstance(g, dict):
            print({k: tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
                   for k, v in g.items()})

print("\n=== STOCHASTIC HELPER SOURCE ===")
print(inspect.getsource(wm_mc_raw146))
print(inspect.getsource(wm_mc145))

print("\n=== RESPONSE ESTIMATES ===")
for filename in ["dynamic_iv.csv", "sensor_iv.csv", "kernel_targets.csv"]:
    df = pd.read_csv("/work/" + filename)
    print(filename, "\n", df.to_string(index=False))
print("\nActive edits:", edited())
```
