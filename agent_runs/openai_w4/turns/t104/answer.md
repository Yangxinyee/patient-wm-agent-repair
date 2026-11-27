The retained fixtures contain factual and controlled-dose posterior inputs, and the action gates were tested on stay-separated data. That gives us a way to test a repair before submitting weights. I’ll now check the innovation-decoder accuracy and posterior dimensions; poor decoding or leakage onto non-drug bundles would rule out this route.

```python
print("=== RAW DOSE-INNOVATION VALIDATION ===")
print(raw_table87.to_string(index=False))
print("\n=== ACTION GATE VALIDATION ===")
print(pd.DataFrame(gate_rows88).to_string(index=False))

print("\n=== LATENT WEIGHTS ===")
for name in NAMES:
    if name.startswith(("core.z_", "core.flow", "core.a_in", "core.delta_in",
                        "core.b_in", "core.c_out", "core.out_norm")):
        print(name, SHAPES[name])

print("\n=== LONGITUDINAL MOMENTS ===")
print("moment_data length", len(moment_data))
for item in moment_data[:3]:
    print(summary91(item))
print("_moment_inputs", _moment_inputs)
print("HX columns", list(HX.columns))
print("panel index", pan.index.names, "example", pan.head(3).to_string())

print("\n=== REPAIR FIXTURE ALIGNMENT ===")
for name in ["idx73", "indices73", "ix73", "idx74", "dataset74", "head_data74",
             "fit82", "test82", "targets82", "eps82", "hidden82",
             "dose_innovation", "innovations", "residuals", "nuisance_models"]:
    if name in globals():
        print(name, summary91(globals()[name]))
print("candidate retained names", [
    n for n in globals() if not n.startswith("_")
    and any(s in n.lower() for s in ["nuis", "innovation", "eps", "dose_model", "dose_pred"])
][:55])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 92\n"
            "Inspecting held-out raw-input innovation regressions and gate margins, "
            "plus exact posterior tensor dimensions. No weights edited.\n")
```
