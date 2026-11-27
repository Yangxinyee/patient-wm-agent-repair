The converged statistical fits suggest stronger accumulated effects than the original model predicts, especially for norepinephrine. I’ll recompute targets for the exact rollout cohort and compare the constrained decoders. A better decoder should provide more dose-edit response per unit of factual residual, while preserving the zero-dose constraint before treatment.

```python
print("=== UPDATED RECORD-BASED TARGETS ===")
targets165 = record_impulse_targets(
    tune_panel, rollout_ids137,
    -1.8, float(latent_ins158["parameters"]["rho"]),
    3.6, float(latent_ne157["parameters"]["rho"])
)
print(type(targets165).__name__)
if isinstance(targets165, pd.DataFrame):
    print(targets165.columns.tolist())
    print(targets165.head(4).to_string(index=False))
    if "edit" in targets165.columns:
        print(targets165.groupby("edit").mean(numeric_only=True).to_string())
elif isinstance(targets165, dict):
    for key165, value165 in targets165.items():
        print(key165, repr(value165)[:500])

print("\n=== DECODER TRADEOFFS ===")
for key165 in sorted(constraint_candidates147):
    print("Candidate", key165)
    report165 = report_decoder147(constraint_candidates147[key165])
    if report165 is not None:
        print(report165)

print("\n=== SUPPORTING FEATURE SHAPES ===")
for name165 in ["F147", "D147", "Y147", "dt147", "active147",
                "Ef132", "V128", "validated_W151", "softdelta128"]:
    if name165 in globals():
        value165 = globals()[name165]
        print(name165, type(value165).__name__, getattr(value165, "shape", None))
```
