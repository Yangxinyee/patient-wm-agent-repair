The normalization test exposed inconsistent cached coefficients, so the reverted candidate was not safe to test. I’ll reconstruct the affine decoder directly from the standardized regression and verify it on all factual prelude features. Matching the regression predictions with negligible conversion error is the required check before reinstalling anything.

```python
print("=== FIT STRUCTURES ===")
for name in ["prelude_fits112", "raw_fits117", "raw_fits119"]:
    store = globals()[name]
    item = store["E"]
    print(name, "E type:", type(item).__name__)
    if isinstance(item, dict):
        for key, value in item.items():
            print(" ", key, type(value).__name__, getattr(value, "shape", None),
                  list(value.keys())[:8] if isinstance(value, dict) else "")
    elif isinstance(item, (tuple, list)):
        print([(type(x).__name__, getattr(x, "shape", None)) for x in item])

print("\n=== CANONICAL STANDARDIZED REGRESSION ===")
for key in ["B", "mu", "sd"]:
    arr = np.asarray(fit128[key])
    print(key, arr.shape, "range", float(arr.min()), float(arr.max()))
Efull134 = np.asarray(prelude_E111, dtype=np.float64)
print("Factual E normalization error:",
      np.max(np.abs(np.sum(Efull134 / gamma133, axis=-1) - constant133)))

B134 = np.asarray(fit128["B"], dtype=np.float64)
mu134 = np.asarray(fit128["mu"], dtype=np.float64)
sd134 = np.asarray(fit128["sd"], dtype=np.float64)
assert mu134.shape == sd134.shape == (192,)
assert B134.shape == (193, 2)
standardized134 = (Efull134 - mu134) / sd134
pred134 = standardized134 @ B134[1:] + B134[0]
raw_coef134 = B134[1:] / sd134[:, None]
affine_b134 = B134[0] - mu134 @ raw_coef134
Wdose134 = raw_coef134 + np.outer(1.0 / gamma133, affine_b134 / constant133)

print("Canonical affine intercept:", affine_b134)
print("Canonical raw/folded coefficient norms:",
      np.linalg.norm(raw_coef134, axis=0), np.linalg.norm(Wdose134, axis=0))
print("Canonical prediction RMS, all slots:",
      np.sqrt(np.mean(pred134**2, axis=(0, 1))))
print("Canonical prediction RMS, mean slot:",
      np.sqrt(np.mean(pred134.mean(axis=1)**2, axis=0)))
print("Standardized vs affine max error:",
      np.max(np.abs(pred134 - (Efull134 @ raw_coef134 + affine_b134))))
print("Standardized vs folded max error:",
      np.max(np.abs(pred134 - Efull134 @ Wdose134)))
print("Previous decoder vs canonical coefficient norms:",
      np.linalg.norm(Edecoder128 - raw_coef134),
      np.linalg.norm(Wdose128 - Wdose134))

with open("/work/notes.md", "a") as f:
    f.write("""
## Runs 133–134: cached-decoder inconsistency
No weights remain edited. The prior Wdose128 failed affine reconstruction on actual E119 too. Reconstructing the decoder directly from fit128 B/mu/sd, with intercept FIRST, then folding its intercept using sum(E/gamma)=sum(beta/gamma). Only a construction matching standardized predictions on all factual E features will be used.
""")
print("Current edits:", edited())
```