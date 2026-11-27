The paired insulin improvements are smaller than their stay-to-stay uncertainty, while the norepinephrine half-dose deterioration is clearer. I’m therefore not treating the blend as an established improvement. I’ll examine whether a gain-only adjustment could strengthen the main insulin readout at comparable factual cost. That hypothesis needs direct factual and rollout validation, not just decoder proxies.

```python
print("COMPACT RECORD-EFFECT EVIDENCE")
for prefix in ("Run 6:", "Run 12:", "Run 38:", "Run 39:",
               "Runs 157", "Run178:", "Runs181"):
    found = [s for s in sections234[1:] if s.startswith(prefix)]
    for s in found:
        print(s[:1250])

print("MATCHED TARGET TABLE")
print("Columns:", record_targets217.columns.tolist())
print(record_targets217.head(4).to_string(index=False))
target_join236 = dose_e235.merge(
    record_targets217, on=["stay_id", "hour", "edit"],
    suffixes=("_model", "_record"), validate="one_to_one"
)
print("Merged columns:", target_join236.columns.tolist())
if "target" in target_join236.columns:
    print(target_join236[target_join236["reached"].astype(bool)].groupby("edit")[
        ["shift", "target"]].mean().to_string())

facts_base236, summ_base236 = paired_factual_233([26, 27])
facts_strong236, summ_strong236 = paired_factual_233([29, 30, 31, 32])
print("BASE-GAIN FACTUAL:", json.dumps(summ_base236))
print("STRONG-GAIN FACTUAL:", json.dumps(summ_strong236))

stay236 = facts_base236["stay_id"].unique()
gain_table236 = []
for label, frame, gi, gn in [
    ("base", facts_base236, 1.0, 1.0),
    ("strong", facts_strong236, 1.2, 1.35),
    ("accepted", factual_base233, 1.2, 1.05),
    ("half_blend", factual233, 1.2, 1.05)
]:
    sub = frame[frame["stay_id"].isin(stay236)]
    gain_table236.append({
        "candidate": label, "insulin_gain": gi, "NE_gain": gn,
        "stays": sub["stay_id"].nunique(),
        "paired_delta_common_stays": float(sub["delta_nll"].mean())
    })
gain_table236 = pd.DataFrame(gain_table236)
print("COMMON-STAY GAIN DIAGNOSTIC")
print(gain_table236.to_string(index=False))

print("EXACT GAIN-ONLY CANDIDATES")
for name in sorted(list(globals())):
    obj = globals()[name]
    if "candidate_B" in name and isinstance(obj, np.ndarray) and obj.shape == (192, 192):
        y = (obj.astype(np.float64) - np.asarray(base218, dtype=np.float64)).reshape(-1)
        gains = np.linalg.lstsq(basis218, y, rcond=None)[0]
        error = float(np.max(np.abs(y - basis218 @ gains)))
        if error < 2e-5:
            print(name, "gains", np.round(gains, 5).tolist(), "mapping residual", error)

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run236: gain-only versus half-blend diagnostics\n")
    f.write("No weights changed. Half-blend insulin improvements have paired stay SEs "
            "larger than their means; NE-half worsening is clearer.\n")
    f.write(gain_table236.to_json(orient="records") + "\n")
    f.write("Gain-only diagnostics are planning proxies; any new candidate requires "
            "full paired factual validation and the exact rollout.\n")
```
