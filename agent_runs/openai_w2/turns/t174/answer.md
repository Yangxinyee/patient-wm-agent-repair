The stronger edit predicts −14.18/+6.78 for insulin and +4.32/−2.07 for norepinephrine. Insulin is near the current record-based target; norepinephrine may still be too weak. I’ll preserve this candidate, inspect factual errors, and revisit the persistence evidence before choosing further gains. Independent-cohort fit and stable effect estimates would support retaining it.

```python
validated_B151 = as_np150(load("core.b_in.weight"))
validated_W151 = candidate_W150.copy()
validated_gains151 = candidate_gains150.copy()

fo151 = pd.read_csv("/work/fwd_16/factual_original.csv")
fe151 = pd.read_csv("/work/fwd_16/factual_edited.csv")
keycols151 = ["stay_id", "hour", "event", "observed"]
assert fo151[keycols151].equals(fe151[keycols151])
audit151 = fo151[keycols151].copy()
audit151["delta_nll"] = fe151["nll"] - fo151["nll"]
audit151["delta_mean"] = fe151["predicted_mean"] - fo151["predicted_mean"]

print("=== FACTUAL AUDIT ===")
print(audit151.groupby("event").agg(
    n=("delta_nll", "size"),
    nll_change=("delta_nll", "mean"),
    mean_shift=("delta_mean", "mean"),
    rms_shift=("delta_mean", lambda x: np.sqrt(np.mean(x*x)))
).to_string())
print("\nPre-ICU / initial predictions:")
print(audit151.loc[audit151.hour < 0].groupby("event").agg(
    n=("delta_nll", "size"),
    nll_change=("delta_nll", "mean"),
    max_abs_shift=("delta_mean", lambda x: np.max(np.abs(x)))
).to_string())
dn151 = audit151.delta_nll.mean()
cluster151 = (audit151.delta_nll - dn151).groupby(audit151.stay_id).sum()
se151 = np.sqrt(len(cluster151)/(len(cluster151)-1)
                * np.sum(cluster151.to_numpy()**2)) / len(audit151)
print("Overall NLL change and stay-cluster SE:", dn151, se151)

print("\n=== EFFECT PERSISTENCE NOTES ===")
note_budget151 = 5200
for section151 in sections149:
    header151 = section151.splitlines()[0] if section151.strip() else ""
    if any(header151.startswith(s) for s in [
        "## Run 12:", "## Run 38:", "## Run 39:"
    ]):
        text151 = section151[:min(len(section151), note_budget151)]
        print(text151)
        note_budget151 -= len(text151)
        if note_budget151 <= 0:
            break

print("\n=== DOSE OUTPUT FORMAT ===")
dose151 = pd.read_csv("/work/fwd_17/dose_edit_edited.csv")
print(dose151.columns.tolist())
print(dose151.head(3).to_string(index=False))

with open("/work/notes.md", "a") as f151:
    f151.write(
        "\n## Forward 16–17 / run 151: initial-safe stronger edit validated locally\n"
        "Candidate key (0.3,0.0001), gains [-10,+12], only core.b_in.weight. "
        "99-stay paired factual NLL 1.57771 vs 1.57474, +0.00297, passes. "
        "Paired exact rollouts: insulin x2 -14.181, half +6.780; "
        "NE x2 +4.316, half -2.067. Insulin is near the earlier record-supported "
        "accumulation estimate; NE remains weaker. Saved exact weights as "
        "validated_B151 and decoder/gains as validated_W151/validated_gains151. "
        "Need independent-cohort factual validation, target/persistence audit, "
        "and possibly a more efficient NE correction before final acceptance.\n"
    )
```
