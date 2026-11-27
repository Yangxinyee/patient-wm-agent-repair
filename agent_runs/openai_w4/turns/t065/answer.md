The immediate estimates support insulin lowering glucose by about 2 units per unit of rate and norepinephrine raising MAP by about 3.2. The lag tables suggest persistence, but their six-hour kernels are still provisional. I’ll inspect the recovered encoding recipes and dose residuals so we can extend the causal estimates and test a narrowly targeted repair; both need independent validation.

```python
print("=== ENCODING RECIPES ===")
for n in ("atom_recipe_repair", "measurement_recipe_repair", "r52_best",
          "repair_ekinds", "repair_units", "repair_rate_units", "head_cfg_repair"):
    print(n, globals().get(n))
print("=== BEST MEASUREMENT TESTS ===")
print(r52_table.sort_values(r52_table.columns[-1]).head(12).to_string(index=False))
print("=== HELPER METADATA ===")
for n in ("atom_embedding_repair", "pre_context_repair", "rp_action_encoder",
          "rp_linear", "rp_norm", "rp_mlp", "head_features_fixed"):
    fn56 = globals().get(n)
    if callable(fn56):
        print(n, "args", fn56.__code__.co_varnames[:fn56.__code__.co_argcount],
              "names", fn56.__code__.co_names,
              "file", fn56.__code__.co_filename)
print("=== RESIDUAL OBJECTS ===")
for drug56 in ("insulin", "norepinephrine"):
    print(drug56)
    for key56, val56 in dml_residuals[drug56].items():
        print(" ", key56, type(val56).__name__, getattr(val56, "shape", None))
    print("models", type(dml_models[drug56]).__name__)
    mi56 = _moment_inputs[drug56]
    print("moment inputs", type(mi56).__name__,
          list(mi56.keys()) if isinstance(mi56, dict) else None)
print("=== STATE FILES ===")
for dir56 in sorted(Path("/work").glob("fwd_*")):
    print(dir56.name, [p.name for p in dir56.iterdir()])
print("=== CORE TENSORS ===")
for name56 in NAMES:
    if name56.startswith("core."):
        print(name56, SHAPES[name56])
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 56\nImmediate innovation estimates: insulin -2.049 +/-0.046, NE +3.171 +/-0.088 per rate; insulin RD broadly agrees. Six-lag kernels show non-single-exponential persistence. Inspecting encoding recipes and residuals for independent causal and repair validation. No weights edited.\n")
```