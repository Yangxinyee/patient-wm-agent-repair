Across all 400 tuning stays, the trial adds 0.00347 nats per reading; its one-sided 95% upper bound is 0.00496, narrowly below the limit. I’ll recover the previous rollout’s exact stays and the record-based effect evidence. The blend is justified only if it improves insulin half-dose predictions enough to warrant its slightly higher factual cost.

```python
dose_orig234 = pd.read_csv("/work/fwd_38/dose_edit_original.csv")
dose_edit234 = pd.read_csv("/work/fwd_38/dose_edit_edited.csv")
print("Previous rollout columns:", dose_edit234.columns.tolist())
print(dose_edit234.head(8).to_string(index=False))
stay_col234 = next(c for c in dose_edit234.columns if "stay" in c.lower())
rollout_ids234 = sorted(dose_edit234[stay_col234].dropna().astype(int).unique().tolist())
print("MATCHED ROLLOUT STAYS:", json.dumps(rollout_ids234))
print("Number of stays:", len(rollout_ids234))

print("README")
print(Path("/data/README.md").read_text())
notes234 = Path("/work/notes.md").read_text()
print("EARLY RECORD-EFFECT EVIDENCE")
print(notes234[:16000])
print("LATER IDENTIFICATION NOTES")
sections234 = notes234.split("\n## ")
matches234 = [
    s for s in sections234[1:]
    if any(t in s.lower() for t in
           ("identif", "instrument", "confound", "causal evidence",
            "persistence", "b_insulin", "b_ne", "negative control"))
]
print("\n\n".join(s[:2200] for s in matches234[-9:]))

print("Current vs candidate_B232:",
      float(np.max(np.abs(current233 - np.asarray(candidate_B232, dtype=np.float32)))))
print("Fallback verification:",
      float(np.max(np.abs(np.asarray(fallback_B232) - np.asarray(validated_B225)))))
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run234: rollout-cohort and causal-evidence recovery\n")
    f.write("No weight changes. Half-dose blend factual delta is 0.00347139, "
            "one-sided 95% upper 0.00496064; fallback delta is 0.00330373.\n")
    f.write("Recovered " + str(len(rollout_ids234)) +
            " matched rollout stays from forward38 for direct response comparison.\n")
```
