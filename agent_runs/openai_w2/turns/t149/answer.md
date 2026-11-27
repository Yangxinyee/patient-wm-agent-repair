The sanity check failed: the feature array I used did not reproduce the decoder’s affine conversion, and its apparent factual displacement was much larger than expected. I’ll revert the candidate before using a forward pass and check the decoder against the actual normalized prelude output. A small conversion error and the previously measured residuals would validate the construction.

```python
drop_edit("core.b_in.weight")

gamma133 = as_numpy132(load_original("core.prelude_norm.weight")).astype(np.float64)
beta133 = as_numpy132(load_original("core.prelude_norm.bias")).astype(np.float64)
constant133 = np.sum(beta133 / gamma133)
Wfold133 = Edecoder128 + np.outer(1.0 / gamma133, bdose128 / constant133)

print("=== NORMALIZATION AND FOLDING ===")
print("Normalization constant:", constant133)
print("Difference from previous folded decoder:",
      np.linalg.norm(Wfold133 - Wdose128))
Etest133 = as_numpy132(E119).astype(np.float64)
print("E119 normalization constraint max error:",
      np.max(np.abs(np.sum(Etest133 / gamma133, axis=-1) - constant133)))
print("E119 previous/new affine conversion errors:",
      np.max(np.abs(Etest133 @ Wdose128 - (Etest133 @ Edecoder128 + bdose128))),
      np.max(np.abs(Etest133 @ Wfold133 - (Etest133 @ Edecoder128 + bdose128))))
print("E119 previous/new residual RMS:",
      np.sqrt(np.mean((Etest133 @ Wdose128)**2, axis=(0, 1))),
      np.sqrt(np.mean((Etest133 @ Wfold133)**2, axis=(0, 1))))

print("\n=== FEATURE FAMILY INVENTORY ===")
for name, obj in sorted(list(globals().items())):
    if isinstance(obj, dict) and any(k in obj for k in ["E", "ffn", "prelude", "factual"]):
        print(name, "keys:", list(obj.keys())[:15])
    elif isinstance(obj, (np.ndarray, torch.Tensor)) and tuple(obj.shape) == (4250, 4, 192):
        print(name, type(obj).__name__, tuple(obj.shape))

print("\n=== SELECTED RECENT VARIABLES ===")
for name in ["family112", "fit128", "qf128", "pp128", "gamma128",
             "beta128", "cv128", "cc128", "shift128", "use128"]:
    obj = globals().get(name)
    if isinstance(obj, dict):
        print(name, "keys:", list(obj.keys()), "report:", obj.get("report"))
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, tuple(obj.shape),
              as_numpy132(obj) if obj.numel() <= 8 else "array"
              if isinstance(obj, torch.Tensor) else obj if obj.size <= 8 else "array")
    elif obj is not None:
        print(name, str(obj)[:300])

with open("/work/notes.md", "a") as f:
    f.write("""
## Run 133: candidate reverted before forward testing
Run 132 used factual112 without confirming its feature family: affine-folding error 1.584 and apparent factual residual RMS 0.797/0.045 contradicted run 128. Reverted core.b_in.weight immediately. Checking normalization invariant against actual E119 and locating the full factual E array; the run-132 candidate is NOT validated.
""")
print("Current edits:", edited())
```