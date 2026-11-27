The earlier tests show that stronger norepinephrine edits broke the factual limit, while the insulin edit passed on two samples. I’ll now match the current rollout to the record-derived targets and select an unused factual-validation group. That will test whether this trial’s narrow pass generalizes beyond the sample used so far.

```python
print("=== CALIBRATION AND SHRINKAGE HISTORY ===")
print(notes221[66997:68225])
print(notes221[68594:69151])
print(notes221[72873:73921])

print("\n=== RECORD-BASED EFFECT EVIDENCE ===")
print(notes221[36953:38688])
print(notes221[40170:41170])

print("\n=== CURRENT ROLLOUT AND MATCHED TARGETS ===")
d222 = pd.read_csv("/work/fwd_25/dose_edit_edited.csv")
print("columns", list(d222.columns))
print(d222.head(2).to_string(index=False))
try:
    matched222 = targets214("/work/fwd_25/dose_edit_edited.csv")
    print("matched record-derived estimates:", matched222)
except Exception as e:
    print("target helper:", type(e).__name__, str(e)[:500])

print("\n=== CALIBRATION HELPER SIGNATURES ===")
import inspect
for k in list(globals()):
    if callable(globals()[k]) and re.search(
        r"(?:decode|build.*static|static.*build|calib|register|head.*prob|prob.*head)", k
    ):
        try:
            print(k, str(inspect.signature(globals()[k])))
        except Exception:
            pass

seen222 = set()
for p in Path("/work").glob("fwd_*/factual_original.csv"):
    seen222.update(pd.read_csv(p, usecols=["stay_id"])["stay_id"].unique().tolist())
all_ids222 = sorted(tuning214["stay_id"].unique().tolist())
fresh222 = [int(s) for s in all_ids222 if s not in seen222][:100]
print("\nUnused tuning stays available:", sum(s not in seen222 for s in all_ids222))
print("NEXT FACTUAL REQUEST:", json.dumps({
    "type": "factual", "model": "both", "split": "tuning", "stays": fresh222
}))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 222: target matching and fresh-validation preparation\n"
            "Reviewing previous shrinkage/calibration outcomes and applying the impulse-model target "
            "to Forward25 reached rows. Selected previously unused tuning stays for a factual "
            "generalization check of trial219. Weights unchanged. Python runs: 222; forwards: 25.\n")
```