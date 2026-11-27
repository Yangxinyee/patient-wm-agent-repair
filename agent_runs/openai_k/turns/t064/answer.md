The full action pathway responds to both drugs, but its immediate slopes are much smaller than the record estimates, and the insulin response saturates at larger multipliers. The value-head features were reconstructed exactly; the later inventory error did not affect those results. I’ll expand the state sample before fitting a correction, so a proposal must work beyond the 64 probes.

```python
print("=== COMPLETE STATE ARRAYS ===")
for k, v in list(globals().items()):
    shp = getattr(v, "shape", None)
    if isinstance(shp, (tuple, torch.Size)) and len(shp) >= 2 and shp[0] == 4035:
        print(k, tuple(shp))
print("\n=== STATE / PANEL CONTAINERS ===")
for k, v in list(globals().items()):
    if isinstance(v, dict) and any(s in k.lower() for s in ("state", "probe", "panel", "imp", "dynamic")):
        print(k, list(v)[:15])
print("\n=== CORE HELPER CONVENTIONS ===")
print("names:", repaired_core_local.__code__.co_names)
print("constants:", str(repaired_core_local.__code__.co_consts)[:900])
print("probe indices:", core_probe_indices[:10])
print("probe mask first row:", core_probe_mask[0, :15].tolist())
print("probe ages first row:", core_probe_ages[0, :8, 0].tolist())
print("probe hours:", rr33_hours[core_probe_indices[:5]])
print("state hours first:", rr33_hours[:12])
print("data columns:", df.columns.tolist())
print("panel candidates:",
      [(k, tuple(v.shape), list(v.columns)[:8]) for k, v in list(globals().items())
       if isinstance(v, pd.DataFrame) and len(v) < 1000000][:8])
with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 55–58: causal pathway audit\n"
            "- Training-record one-hour effects: insulin/glucose about -2 per rate unit; "
            "NE/arterial MAP about +3.05. Glucose impulses persist longer than MAP.\n"
            "- Existing posterior-coordinate directions are too weak/heterogeneous for a safe direct edit.\n"
            "- Exact reconstructed value-head hidden features. Complete-path immediate slopes on 64 probes: "
            "insulin x2 -0.342 (finger), x3 -0.178; NE x2 +0.249 (arterial).\n"
            "- No weights edited. Expanding state coverage before a preservation-constrained head correction.\n"
            "- Python runs: 58; forward passes: 4.\n")
```
