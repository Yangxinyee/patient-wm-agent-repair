The history-adjusted estimates agree on a positive norepinephrine effect of roughly +3.5 to +3.9 per rate unit. Insulin estimates span about −1.55 to −1.97. Persistence remains the main uncertainty. I’ll recover the latent-fit details and compare dose decoders; a useful replacement must increase intervention sensitivity without increasing factual dose noise.

```python
import inspect

print("=== LATENT-STATE EVIDENCE ===")
for section164 in sections160:
    if section164.startswith(("## Run 38:", "## Run 39:")):
        print(section164[:4000])

print("\n=== LOCAL FUNCTIONS ===")
functions164 = []
for name164, obj164 in list(globals().items()):
    if inspect.isfunction(obj164) and obj164.__module__ == "__main__":
        try:
            sig164 = str(inspect.signature(obj164))
        except Exception:
            sig164 = "(signature unavailable)"
        functions164.append(name164 + sig164)
print("\n".join(sorted(functions164))[:3300])

print("\n=== CONSTRAINED DECODER CANDIDATES ===")
for key164, obj164 in constraint_candidates147.items():
    if isinstance(obj164, dict):
        print(key164, "dict", list(obj164.keys()))
    else:
        print(key164, type(obj164).__name__, getattr(obj164, "shape", None))

print("\n=== LATEST FIT OBJECT NAMES ===")
for name164 in sorted(list(globals())):
    if name164.endswith(("156", "157", "158")):
        obj164 = globals()[name164]
        if isinstance(obj164, dict):
            print(name164, "dict", list(obj164.keys())[:12])
        elif isinstance(obj164, (np.ndarray, pd.DataFrame)):
            print(name164, type(obj164).__name__, obj164.shape)
        elif isinstance(obj164, (float, int)):
            print(name164, obj164)
```
