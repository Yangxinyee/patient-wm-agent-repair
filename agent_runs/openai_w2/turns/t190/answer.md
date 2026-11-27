The decoder variants offer only small improvements in norepinephrine sensitivity. Also, the target averages just printed included untreated hours, unlike the judged readout. I’ll align the targets with the rollout’s eligibility mask and inspect factual errors before changing gains. A persistent norepinephrine shortfall after that alignment would justify a stronger candidate.

```python
dose_orig166 = pd.read_csv("/work/fwd_17/dose_edit_original.csv")
dose_edit166 = pd.read_csv("/work/fwd_17/dose_edit_edited.csv")
print("Rollout edit labels:", sorted(dose_edit166["edit"].unique()))
print("Rollout columns:", dose_edit166.columns.tolist())

aliases166 = {
    "norepinephrine_x2": "ne_x2",
    "norepinephrine_half": "ne_half",
    "norepinephrine_x0.5": "ne_half",
    "insulin_x0.5": "insulin_half"
}
for df166 in [dose_orig166, dose_edit166]:
    df166["edit"] = df166["edit"].replace(aliases166)

aligned166 = dose_edit166.merge(
    targets165, on=["stay_id", "hour", "edit"], how="left", validate="one_to_one"
).merge(
    dose_orig166[["stay_id", "hour", "edit", "shift"]].rename(
        columns={"shift": "original_shift"}
    ),
    on=["stay_id", "hour", "edit"], validate="one_to_one"
)
print("\n=== EXACT-COHORT, REACHED-HOUR COMPARISON ===")
for edit166, group166 in aligned166.groupby("edit"):
    use166 = group166[group166["reached"].astype(bool)]
    staymeans166 = use166.groupby("stay_id")[["original_shift", "shift", "target"]].mean()
    print(edit166, "stays", len(staymeans166), "hours", len(use166),
          "per-stay means", staymeans166.mean().round(5).to_dict(),
          "pooled-hour means", use166[["original_shift", "shift", "target"]].mean().round(5).to_dict(),
          "missing targets", int(use166["target"].isna().sum()))

print("\n=== FACTUAL FILE FORMAT ===")
fact_o166 = pd.read_csv("/work/fwd_16/factual_original.csv")
fact_e166 = pd.read_csv("/work/fwd_16/factual_edited.csv")
print(fact_o166.columns.tolist())
print(fact_o166.head(3).to_string(index=False))
print("\nCurrent edit matches saved fallback:",
      np.allclose(np.asarray(load("core.b_in.weight")), np.asarray(validated_B151)))
```
