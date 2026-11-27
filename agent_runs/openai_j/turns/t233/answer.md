The local test shows that the edit strengthens norepinephrine’s immediate MAP response only modestly, so rollout feedback is not the whole problem. The current weights also exactly match the saved calibrated candidate. I’ll recover the separate half-dose and double-dose contrast designs to test a correction aimed more directly at the weak double-dose response. A larger response with little additional factual loss would support that approach.

```python
print("=== CONTRAST-RELATED VARIABLES ===")
for n207, o207 in list(globals().items()):
    if any(s in n207 for s in ("146", "172")) and not inspect.isfunction(o207):
        if isinstance(o207, dict):
            print(n207, "dict keys", list(o207)[:16])
        elif hasattr(o207, "shape"):
            print(n207, type(o207).__name__, tuple(o207.shape))
        elif isinstance(o207, (str, int, float, list, tuple)):
            print(n207, repr(o207)[:140])

print("\n=== COUNTERFACTUAL CONTAINER DISCOVERY ===")
for n207, o207 in list(globals().items()):
    if isinstance(o207, dict) and n207 not in ("__builtins__", "SHAPES", "VOCAB"):
        matches207 = [
            k for k, v in o207.items()
            if isinstance(v, dict) and "inputs" in v and "delta_rate" in v
        ]
        if matches207:
            print(n207, "contrast keys:", matches207[:20])

print("\n=== STATE/CONTRAST BUILDING HELPERS ===")
for n207, o207 in list(globals().items()):
    if inspect.isfunction(o207) and any(
        s in n207 for s in ("cf", "counter", "mc", "replay", "state", "head_input")
    ):
        print(n207, inspect.signature(o207), o207.__code__.co_names[:25])

print("\n=== FURTHER CAUSAL EVIDENCE ===")
p207 = notes204.find("## Sensor-proxy dynamic check")
print(notes204[p207:p207+4200])

print("\n=== GATING AND SPLIT ===")
print("Free MAP neurons:", head177_free)
print("Ordinal logit direction:", np.asarray(wm_v151))
print("Gate shape:", np.asarray(trial158_gate).shape)
print("Fit/CV stays:",
      head_meta.loc[cal190_fitmask, "stay_id"].nunique(),
      head_meta.loc[cal190_cvmask, "stay_id"].nunique())
```
