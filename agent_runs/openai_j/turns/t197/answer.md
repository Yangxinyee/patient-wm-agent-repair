The MAP addition leaves the local insulin response unchanged and has a local tuning loss increase of +0.00401 nats, within the +0.005 limit. Its local norepinephrine response is still modest. I’ll recover the exact validation IDs, then check GPU factual loss before testing rollout amplification.

```python
print("=== FIRST VALIDATION COHORT ===")
cohort178 = pd.read_csv("/work/fwd_15/factual_original.csv")["stay_id"].drop_duplicates().tolist()
print([int(x) for x in cohort178])
print("=== SECOND VALIDATION COHORT ===")
cohort178b = pd.read_csv("/work/fwd_17/factual_original.csv")["stay_id"].drop_duplicates().tolist()
print("second cohort size:", len(cohort178b),
      "first/last:", int(cohort178b[0]), int(cohort178b[-1]))
print("=== ROLLOUT FILE SCHEMA ===")
print(pd.read_csv("/work/fwd_16/dose_edit_edited.csv").columns.tolist())

with open("/work/notes.md", "a") as f:
    f.write("- Run 177 local validation: compression logit RMS 0.00047 train/0.00057 tuning; "
            "overall value dNLL +0.00351 train/+0.00401 tuning.\n"
            "- Local insulin response unchanged; NE arterial local slopes half=1.0848, double=0.4531. "
            "Whether this gives useful rollout amplification remains untested.\n")
```
