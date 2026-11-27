The full state-update route is less sensitive than the prior-to-head route, so a large posterior edit could be unsafe. I’ll check the response directions’ signs and units, then test the preservation-only weights locally. That separates reconstruction errors from errors introduced by a causal correction.

```python
import numpy as np
import torch

print("=== RESPONSE DIRECTIONS ===")
for jname, dname in [("J56", "D56_candidates"), ("J65", "D65_candidates")]:
    J = globals().get(jname)
    ds = globals().get(dname)
    if J is not None and isinstance(ds, dict):
        J = np.asarray(J)
        print(jname, J.shape)
        for lam in [0.001, 0.01, 0.1]:
            if lam in ds:
                D = np.asarray(ds[lam])
                print(dname, lam, "mean physical response:\n",
                      np.round(np.einsum("...k,kd->...d", J, D).mean(axis=0), 4))
print("Y65 first eight rows:\n", np.asarray(Y65)[:8])

print("\n=== PRESERVATION WEIGHTS ===")
for n, t in post66_base_weights.items():
    a = torch.as_tensor(t, dtype=torch.float32)
    b = torch.as_tensor(load_original(n), dtype=torch.float32)
    print(n, tuple(a.shape), "max original difference",
          float((a-b).abs().max()), "difference norm", float((a-b).norm()))
print("Clone regression:", post66_clone.coef_.shape,
      np.asarray(post66_clone.intercept_).shape)
print("Rate decoder example:", np.round(np.asarray(post66_rate_models[1.0])[-1], 4))
print("Rate/innovation first rows:\n",
      np.c_[rate66[:8], rtrue66[:8]])
print("W0 type:", type(W0).__name__)

print("\n=== SMALL EXACT RECONSTRUCTION ===")
idx70 = np.flatnonzero(np.any(rate66 > 0, axis=1))[:8]
out70, h70 = r59_core(idx70)
features70 = r59_features(out70, h70, idx70)
print("probe indices:", idx70)
print("out/h shapes:", tuple(out70.shape), tuple(h70.shape))
if isinstance(features70, dict):
    print("feature keys:", list(features70))
else:
    print("features type:", type(features70).__name__,
          "shape:", getattr(features70, "shape", None))
print("r59_features dependencies:", r59_features.__code__.co_names)

print("\n=== RETAINED ARRAYS ===")
for n in sorted(list(globals())):
    if n.endswith(("56", "59", "63")):
        a = globals()[n]
        if hasattr(a, "shape"):
            print(n, tuple(a.shape))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 70: candidate construction audit\n"
            "- Checking physical response signs, clone tensor differences, and exact local head interface.\n"
            "- Will distinguish a preservation-only edit from the subsequent causal correction.\n"
            "- Python runs: 70; forward passes: 4.\n")
```