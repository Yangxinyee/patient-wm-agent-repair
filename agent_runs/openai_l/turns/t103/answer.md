The validated latent probe has essentially no useful effect, even with a large perturbation. I’ll abandon that route and inspect the value head’s inputs, before its nonlinear compression. A useful correction must distinguish changed doses while staying near zero on factual records; otherwise it risks worsening factual fit.

```python
print("=== REMAINING CAUSAL EVIDENCE ===")
print(Path("/work/notes.md").read_text()[5000:10200])

print("\n=== VALUE DECODER ===")
show91(decode67, 3400)

print("\n=== VALUE HEAD TENSORS ===")
for name in NAMES:
    if name.startswith("attr_heads.value_bin"):
        print(name, SHAPES[name])

print("\n=== LATENT PROBE DIAGNOSTICS ===")
print("Maximum absolute finite-difference sensitivity:", np.abs(G93).max())
for name in ("z_in.weight", "core.prelude.0.self_attn.in_proj_weight"):
    tensor = load_original(name)
    print(name, "RMS", float(np.sqrt(np.mean(np.asarray(tensor)**2))))

print("\n=== DECODER CONTEXT CONSTANTS ===")
for name in ("const_train67", "trans_train67", "centres_train67",
             "epsread73", "ctx_fact80"):
    value = globals()[name]
    print(name, type(value).__name__, tuple(value.shape))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 95 / reject latent route\n"
            "Correctly reconstructed latent probes have essentially "
            "zero reading sensitivity. Unit insulin direction cannot "
            "be fitted; NE direction norm=2022 still yields no useful "
            "response. Reject posterior edit. Investigating value-head "
            "input features and a small analytic nonlinear head edit. "
            "No tensors edited.\n")
```