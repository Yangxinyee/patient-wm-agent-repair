The records show negative insulin effects near dose thresholds, but they also show treatment-related observation changes, so the raw associations need caution. The recovery output was truncated before the cached edit definitions. I’ll retrieve those narrowly; the validated edit remains unchanged.

```python
import re

print("RECENT CACHE")
keys229 = [
    k for k in sorted(list(globals()))
    if any(tag in k for tag in ("227", "226", "225", "218", "212"))
]
for k in keys229:
    o = globals()[k]
    if isinstance(o, np.ndarray):
        desc = f"array {o.shape}"
    elif hasattr(o, "shape"):
        desc = f"{type(o).__name__} {tuple(o.shape)}"
    elif isinstance(o, dict):
        desc = "dict " + repr(list(o)[:12])
    elif callable(o):
        desc = "function"
    else:
        desc = repr(o)[:140]
    print(k, desc)

print("\nEVIDENCE SECTION HEADINGS")
heads229 = re.findall(r"^## .*$", notes228, flags=re.M)
selected229 = [
    h for h in heads229
    if re.search(r"effect|persist|latent|identif|record|threshold|causal", h, re.I)
]
print("\n".join(selected229[:45]))

print("\nPERSISTENCE EVIDENCE EXCERPTS")
matches229 = list(re.finditer(r"Kalman|latent persistence|rho_ins|rho_NE|b_insulin", notes228))
for m in matches229[:5]:
    print(notes228[max(0, m.start()-180):m.start()+650])
```
