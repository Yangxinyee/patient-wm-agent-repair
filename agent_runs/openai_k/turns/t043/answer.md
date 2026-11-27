The recovered impulse estimates suggest insulin effects persist for several hours, while MAP effects decay faster. That supports a stronger accumulated response, but only if an edit preserves those dynamics and leaves factual predictions intact. I’ll inspect the validated helpers and saved states to build a dose-centered correction rather than amplify the unreliable features.

```python
import inspect

print("=== RELEVANT HELPERS ===")
for rr38_name in ["repaired_core_local", "rr35_scaled_action_encoding",
                  "embed_record_audit", "encode_sets_audit",
                  "lin0", "norm0", "mlp0"]:
    rr38_obj = globals().get(rr38_name)
    print(rr38_name, inspect.signature(rr38_obj) if callable(rr38_obj) else type(rr38_obj))
    if rr38_name in ["repaired_core_local", "rr35_scaled_action_encoding"]:
        try:
            print(inspect.getsource(rr38_obj))
        except Exception as rr38_exc:
            print("source unavailable:", str(rr38_exc))
print("\n=== KEY OBJECT SHAPES ===")
for rr38_name in ["S", "P", "R", "W0", "post_inputs_audit", "post_hidden_audit",
                  "contexts_audit", "all_u_audit", "all_e_audit",
                  "bundle_records_audit", "causal_positions_audit",
                  "dose_resid", "dose_pred", "dynamic_ir", "dml_results"]:
    rr38_obj = globals().get(rr38_name)
    print(rr38_name, type(rr38_obj), getattr(rr38_obj, "shape", None))
    if isinstance(rr38_obj, dict):
        print("keys:", list(rr38_obj)[:18])
print("\n=== SELECTED TENSORS ===")
for rr38_name in NAMES:
    if rr38_name.startswith(("z_post.", "z_prior.", "z_in.", "z_head.",
                             "attr_heads.value_bin.", "query_norm.",
                             "last_numeric.", "gap.", "out_norm.")):
        print(rr38_name, SHAPES[rr38_name])
print("\n=== REMAINING HELPER NAMES ===")
print(", ".join(k for k in sorted(globals())
                if k.startswith(("repaired", "rr", "state", "ts", "train", "tuning", "value", "world", "z_")))[-2000:])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 38: prepare a dose-centered posterior correction\n"
            "- Record-derived insulin impulses persist for several hours; MAP impulses decay faster.\n"
            "- Inspecting validated local update helpers and saved posterior inputs before implementing a correction.\n"
            "- Python runs: 38; forward passes: 4; no weights edited.\n")
```