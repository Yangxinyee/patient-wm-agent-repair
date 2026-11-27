The saved fits include a linear dose-residual decoder that can be inserted into `core.b_in`. I’ll check its units, held-out errors, and correspondence to the finite-difference measurements before choosing an edit strength. This route is promising only if dose-edit signals are substantially larger than its factual residuals.

```python
print("=== RECORD-DERIVED EFFECT NOTES ===")
lines123 = Path("/work/notes.md").read_text().splitlines()
selected123 = [s for s in lines123 if any(t in s.lower() for t in (
    "clean threshold", "rich history", "record-derived", "provisional",
    "next-hour effect", "autoregress", "persistence", "g-computation",
    "estimated rollout", "rollout targets", "record target"
))]
print("\n".join(selected123)[:2400])

print("\n=== RESIDUAL FIT REPORTS ===")
for fam123, fits123 in prelude_fits112.items():
    for g123, fit123 in fits123.items():
        print(fam123, g123, fit123["report"])
print("raw E fit keys:", raw_fits119["E"].keys())
print("E linear norms:", np.linalg.norm(E_linear_dose119, axis=0))
print("affine/linear max error:",
      np.max(np.abs(affine117 - linear117)))

print("\n=== BUNDLE AND HEAD METADATA ===")
print("sm_repair columns:", list(sm_repair.columns))
print(sm_repair.head(2).to_string(index=False))
print("probe_head110:", type(probe_head110).__name__)
if hasattr(probe_head110, "columns"):
    print(list(probe_head110.columns))
    print(probe_head110.head(3).to_string(index=False))
elif isinstance(probe_head110, dict):
    print({k: (v.shape if hasattr(v, "shape") else repr(v)[:140])
           for k, v in probe_head110.items()})
print("core_rows104:", np.asarray(core_rows104).shape,
      np.asarray(core_rows104)[:5])
print("J120 row norms:", np.linalg.norm(J120, axis=1)[:6])
print("repair_head_cfg:", repair_head_cfg)

print("\n=== AVAILABLE DATA/FIT IDENTIFIERS ===")
names123 = [k for k, v in list(globals().items())
            if not k.startswith("_") and not callable(v)
            and any(s in k.lower() for s in (
                "dose", "target", "mask", "split", "train", "test",
                "drug", "probe", "slope", "effect", "panel"
            ))]
print(", ".join(names123)[-2400:])
```
