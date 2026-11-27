The reconstructed code exposes both the dose encoder and the latent-to-state map. I’ll check those interfaces before editing: the next candidate should transmit a dose innovation into the recurrent state, rather than rely on the weak prior bridge. It will be useful only if its factual shifts align with observed prediction errors, not merely if it moves predictions.

```python
print("FOCUSED IMPLEMENTATION")
for name85 in ["r59_core", "dataset80", "prior_features74", "bridge_frontier82", "repaired_core_local"]:
    fn85 = globals()[name85]
    print("\n", name85)
    try:
        print(inspect.getsource(fn85))
    except Exception:
        print("globals used:", fn85.__code__.co_names)
        print("constants:", fn85.__code__.co_consts)
print("\nLATENT / SENSITIVITY OBJECTS")
for k85 in sorted(list(globals())):
    if any(s in k85 for s in ["J56", "J65", "D65", "Wpost", "Wp", "w66", "policy", "impulse", "dynamic", "expected", "target_shifts"]):
        v85 = globals()[k85]
        if hasattr(v85, "shape"):
            print(k85, tuple(v85.shape))
            if v85.size if isinstance(v85, np.ndarray) else v85.numel() <= 32:
                print(str(v85)[:600])
        elif isinstance(v85, dict):
            print(k85, list(v85)[:10])
        elif callable(v85):
            print(k85, "function")
print("\nCARRIER CONSTRUCTION")
print("reserve66:", reserve66)
print("units:", dose_units74)
print("carrier74:\n", carrier74)
print("post66_rate_models keys:", post66_rate_models.keys())
for k85,v85 in post66_rate_models.items():
    print(k85, type(v85), str(v85)[:350])
start85 = notes83.find("## Run 39")
print("\nNOTES 39 onward")
print(notes83[start85:start85+2200])
```
