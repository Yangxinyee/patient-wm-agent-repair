The retained fits can decode changed doses much better than the original posterior features, which saturate for insulin and barely respond to norepinephrine. I’ll check the validated core’s accuracy and the record-derived rollout targets next. If both are usable, a linear dose channel with condition-based centering should be more promising than amplifying the existing signal.

```python
print("=== CORE RECONSTRUCTION ACCURACY ===")
for k in ["rr34_h_err", "rr34_out_err", "rr39_best_alpha"]:
    print(k, globals().get(k))
print("\n=== RELEVANT TENSORS ===")
for name in NAMES:
    if any(s in name for s in [
        "z_post", "z_prior", "z_in", "z_head", "attr_heads.value",
        "query_norm", "type_emb", "last_numeric", "last_status",
        "monitor_head", "count_head"
    ]):
        print(name, SHAPES[name])

print("\n=== CORRECTION FITS ===")
print("decoder coefficients/intercept:",
      rr39_action_decoder.coef_.shape, rr39_action_decoder.intercept_)
for k, fit in rr41_center_models.items():
    print("center penalty", k, "type", type(fit).__name__)
    if hasattr(fit, "coef_"):
        print("  coef/intercept shapes:", fit.coef_.shape,
              np.shape(fit.intercept_))
    elif isinstance(fit, dict):
        print("  keys:", list(fit))
print("rr41 D first rows:", rr41_D[:5])
print("rr41 phi first rows:", rr41_phi[:5])
print("rr41 innovation first rows:", rr41_innov[:5])
print("rr41 expected first rows:", rr41_expected[:5])
print("probe shapes:", core_probe_h.shape, core_probe_e.shape,
      core_probe_u.shape, core_probe_z.shape)
print("posterior inputs/hidden:", post_inputs_audit.shape,
      post_hidden_audit.shape)
print("W0:", type(W0).__name__, "entries", len(W0))

print("\n=== DYNAMIC TARGET AND CORE VALIDATION NOTES ===")
lines47 = Path("/work/notes.md").read_text().splitlines()
section47 = ""
for line in lines47:
    if line.startswith("## "):
        section47 = line
    if any(f"Run {r}" in section47 or f"Runs {r}" in section47
           for r in [6, 7, 8, 9, 28, 29, 30, 31, 32, 33, 34]):
        if len(line) < 1300:
            print(line)
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 47: select dose-channel surgery\n"
            "- Existing posterior dose features saturate; direct action decoder covers multiple multipliers.\n"
            "- Checking validated core error and record-derived aggregate targets before reserving linear dose channels.\n"
            "- Python runs: 47; forward passes: 4; no tensors edited.\n")
```