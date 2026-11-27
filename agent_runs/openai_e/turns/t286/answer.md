The architecture confirms that the fitted detector can be implemented in the prelude’s final feed-forward weights, without gradients. I’ll now recover its coefficient convention and the previously tested MAP-response directions. If the mapping is sound, a small edit should increase norepinephrine sensitivity without disturbing admission states or insulin responses.

```python
import re
notes246 = open("/work/notes.md").read()
print("RELEVANT PRIOR EXPERIMENTS")
sections246 = re.split(r"(?=^## )", notes246, flags=re.M)
for tag in ("Run 81:", "Run 82:", "Run 93:", "Run 94:", "Run 103:", "Run 151:"):
    sec = next((s for s in sections246 if tag in s[:180]), "")
    print(sec[:1100] if sec else "Not found: " + tag)

print("\nDETECTOR CONTRACTS")
for k in ((100.0, .001), (10.0, .001)):
    v = fits244[k]
    print(k, type(v).__name__)
    if isinstance(v, dict):
        for kk, vv in v.items():
            print(" ", kk, type(vv).__name__, getattr(vv, "shape", None),
                  str(vv)[:180] if np.size(vv) <= 8 else "")
    elif isinstance(v, (list, tuple)):
        print(" entries", [(type(x).__name__, getattr(x, "shape", None)) for x in v[:6]])
    else:
        print(" shape", getattr(v, "shape", None))
for k in D244:
    print(k, "D", D244[k].shape, "Y", Y244[k].shape,
          "Y range/mean/RMS", np.min(Y244[k]), np.max(Y244[k]),
          np.mean(Y244[k]), np.sqrt(np.mean(Y244[k]**2)))
print("scale range", scale244.min(), scale244.max(),
      "F RMS", np.sqrt(np.mean(F244**2)), "dose2", dose2_244)

print("\nDIRECTIONS")
for k in ("ne_direction81", "ne_direction82", "ne_direction103"):
    a = globals()[k]
    print(k, a.shape, "norm", np.linalg.norm(a), "range", a.min(), a.max())

print("\nAVAILABLE LOCAL HELPERS")
for k, v in list(globals().items()):
    if callable(v) and hasattr(v, "__code__") and getattr(v, "__module__", "") == "__main__":
        print(k, v.__code__.co_varnames[:v.__code__.co_argcount])

print("\nRELATED ARRAY CONTRACTS")
for k, v in list(globals().items()):
    if not k.startswith("_") and any(k.endswith(str(i)) for i in (81, 82, 93, 94, 103)):
        if hasattr(v, "shape"):
            print(k, v.shape)
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 246: detector-to-weight mapping audit\n"
            "Confirmed the nonlinear route uses prelude FFN output weights (192x768) "
            "and bias. Recovering detector scaling, output directions and prior nonlinear "
            "trial outcomes before editing. Joint236 unchanged. "
            "Python runs: 246; forwards: 29.\n")
```
