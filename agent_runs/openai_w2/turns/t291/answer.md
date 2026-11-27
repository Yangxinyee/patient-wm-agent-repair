The gain-only checks suggest that lowering norepinephrine’s gain can offset some factual cost of strengthening insulin. I’ll test insulin gain 1.40 and norepinephrine gain 1.00, removing the half-dose blend. This may bring the main readout closer to the record-based estimate, but remains provisional: excessive factual loss or poorer rollouts would refute it and trigger restoration of the validated fallback.

```python
import torch

fallback_B237 = np.asarray(validated_B225, dtype=np.float32).copy()
original_B237 = np.asarray(load_original("core.b_in.weight"), dtype=np.float64)
assert np.max(np.abs(original_B237 - np.asarray(base218))) < 2e-6

ins_gain237, ne_gain237 = 1.40, 1.00
candidate_B237 = (
    original_B237
    + ins_gain237 * np.asarray(delta_ins218, dtype=np.float64)
    + ne_gain237 * np.asarray(delta_ne218, dtype=np.float64)
).astype(np.float32)

# This three-point statistical extrapolation is a planning proxy, not validation.
gain_rows237 = gain_table236[gain_table236["candidate"].isin(["base", "strong", "accepted"])]
gain_design237 = np.column_stack([
    np.ones(len(gain_rows237)),
    gain_rows237["insulin_gain"].to_numpy() ** 2,
    gain_rows237["NE_gain"].to_numpy() ** 2
])
gain_coef237 = np.linalg.solve(
    gain_design237, gain_rows237["paired_delta_common_stays"].to_numpy()
)
proxy237 = float(np.array([1., ins_gain237**2, ne_gain237**2]) @ gain_coef237)
print("Common-cohort factual proxy (not validation):", proxy237)

recovered_gains237 = np.linalg.lstsq(
    basis218,
    (candidate_B237.astype(np.float64) - np.asarray(base218)).reshape(-1),
    rcond=None
)[0]
print("Recovered gains:", recovered_gains237)
assert np.max(np.abs(recovered_gains237 - [ins_gain237, ne_gain237])) < 1e-5
assert np.isfinite(candidate_B237).all()
assert candidate_B237.shape == tuple(SHAPES["core.b_in.weight"])

save_edit("core.b_in.weight", torch.from_numpy(candidate_B237.copy()))
assert set(edited()) == {"core.b_in.weight"}
installed_B237 = np.asarray(load("core.b_in.weight"), dtype=np.float32)
assert np.array_equal(installed_B237, candidate_B237)
print("Installed provisional gain-only candidate; hash:",
      hashlib.sha256(installed_B237.tobytes()).hexdigest()[:10])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run237: final gain-only controlled trial installed\n")
    f.write("Only core.b_in.weight edited: original + 1.40*delta_ins218 "
            "+ 1.00*delta_ne218. Half-dose blend removed.\n")
    f.write("Exact accepted fallback is fallback_B237 = validated_B225. "
            "candidate_B232 remains saved but is not the fallback.\n")
    f.write("Three-point additive quadratic common-cohort factual proxy: "
            + str(proxy237) + "; this is extrapolation, not acceptance evidence.\n")
    f.write("Require all-400 paired factual validation and exact matched rollouts. "
            "Record-based drug slopes and persistence remain assumption-dependent estimates.\n")
```
