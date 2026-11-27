The residual fits respond to dose changes, but their factual error is not negligible—especially for insulin. That argues for a small, reversible edit rather than full amplification. The inspection stopped on a tuple-layout mismatch; I’ll resolve that and recover the readout mapping before testing a conservative candidate.

```python
print("=== LINEAR FIT LAYOUT ===")
for fam124 in ("E", "ffn"):
    obj124 = raw_fits119[fam124]
    print(fam124, describe115(obj124))
print("E_linear_dose119:", E_linear_dose119.shape,
      "column norms", np.linalg.norm(E_linear_dose119, axis=0))
print("affine117/linear117 max error:",
      float(np.max(np.abs(affine117 - linear117))))

print("\n=== READOUT ALIGNMENT ===")
print("sm_repair columns:", list(sm_repair.columns))
print(sm_repair.head(2).to_string(index=False))
print("core_rows104:", np.asarray(core_rows104)[:8])
print("probe_head110:", describe115(probe_head110))
if hasattr(probe_head110, "columns"):
    print(list(probe_head110.columns))
    print(probe_head110.head(3).to_string(index=False))
print("repair_head_cfg:", repair_head_cfg)
print("J120:", J120.shape)
print("J120 norms:", np.linalg.norm(J120, axis=1)[:8])
print("delta119:", delta119.min().item(), delta119.max().item())

print("\n=== PERSISTENCE AND TARGET RESULTS ===")
text124 = Path("/work/notes.md").read_text()
for marker124 in ("## Run 12:", "## Run 38:", "## Run 39:"):
    pos124 = text124.find(marker124)
    if pos124 >= 0:
        end124 = text124.find("\n## ", pos124 + 4)
        print(text124[pos124:(end124 if end124 >= 0 else pos124 + 1700)][:1700])

print("\n=== DOSE/DATA IDENTIFIERS ===")
print(", ".join(k for k, v in list(globals().items())
      if not k.startswith("_") and not callable(v)
      and any(s in k.lower() for s in
              ("dose", "target", "mask", "panel", "effect", "slope", "probe")))
      [-1800:])
```
