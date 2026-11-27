The improvement survives stay-level cross-validation: factual variation is about one-third lower at matched dose sensitivity, with similar cross-drug leakage. I’ll now test a candidate with roughly twice the current norepinephrine dose sensitivity, leaving insulin unchanged. It remains provisional: an exact factual NLL increase above 0.005 would reject it.

```python
assert np.array_equal(load("core.b_in.weight"), validated_B151)
assert edited() == ["core.b_in.weight"]

entry173 = V128.astype(np.float64) / delta_scale150[:, None]
original_B173 = np.asarray(load_original("core.b_in.weight"), dtype=np.float64)
reconstructed173 = (
    original_B173
    + entry173 @ (validated_gains151[:, None] * validated_W151.T)
)
error173 = float(np.max(np.abs(reconstructed173 - validated_B151)))
print("Validated-weight reconstruction error:", error173)
assert np.allclose(reconstructed173, validated_B151, atol=3e-5, rtol=1e-6)

candidate_key173 = (0.03, 1e-6)
candidate_strength173 = 2.0
candidate_W173 = validated_W151.copy()
zero_W173 = zero_fact_decoders171[candidate_key173]
zero_metrics173 = report_decoder147(zero_W173)
ne_scale173 = (
    candidate_strength173 * reference171["ne_x2_gain"]
    / zero_metrics173["ne_x2_gain"]
)
candidate_W173[:, 1] = zero_W173[:, 1] * ne_scale173
candidate_gains173 = validated_gains151.copy()
candidate_B173 = (
    original_B173
    + entry173 @ (candidate_gains173[:, None] * candidate_W173.T)
).astype(np.float32)

print("NE decoder scale:", ne_scale173)
print("Candidate diagnostics:", report_decoder147(candidate_W173))
print("Insulin decoder unchanged:",
      np.array_equal(candidate_W173[:, 0], validated_W151[:, 0]))
save_edit("core.b_in.weight", candidate_B173)
print("Installed:", edited())

cohort173 = sorted(
    pd.read_csv("/work/fwd_16/factual_original.csv").stay_id.unique().tolist()
)
assert len(cohort173) <= 100
print("Factual-test stay IDs:", cohort173)

with open("/work/notes.md", "a") as f173:
    f173.write(
        "\n## Runs 170–173: minimum-disturbance NE decoder\n"
        "Reconstructed factual/probe designs exactly. Zero-factual-target "
        "ridge regression, retaining initial-state zero constraints, reduces "
        "matched NE RMS from 0.03311 to about 0.02217 in 5-fold stay CV "
        "(key 0.03,1e-6). Metadata alignment verified against both drug flags. "
        "Installed provisional core.b_in candidate173: insulin unchanged, "
        "NE decoder scaled to twice validated doubled-dose probe sensitivity; "
        "gains remain [-10,+12]. Exact validated_B151 is the fallback. "
        "Require paired factual increase <=0.005 before rollout; "
        "record-based persistence targets remain assumption-dependent.\n"
    )
```
