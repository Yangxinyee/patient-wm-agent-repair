The recovered analyses disagree substantially, so I won’t treat any single regression as the true drug effect. The rollout file also needs the evaluation’s stay-weighted aggregation, not a mean over all rows. I’ll read the saved summary and the later identification conclusions. If the sharper gate weakens the insulin response, I’ll reject it despite its acceptable first-batch NLL.

```python
import json
print("=== FORWARD 35 EXACT SUMMARY ===")
print(json.dumps(json.loads(Path("/work/fwd_35/summary.json").read_text()), indent=2))
print("=== STAY-WEIGHTED REACHED-HOUR SHIFTS ===")
for model_name in ["original", "edited"]:
    d = pd.read_csv(f"/work/fwd_35/dose_edit_{model_name}.csv")
    print(model_name,
          d.loc[d["reached"]].groupby(["edit", "stay_id"])["shift"]
           .mean().groupby("edit").mean().to_dict())

print("=== CHECKPOINT AUDIT ===")
for label, cp in [("sharp240", sharp240_checkpoint),
                  ("validated_gated8", validated_gated8_checkpoint)]:
    print(label, {n: float(np.max(np.abs(
        np.asarray(load(n)) - np.asarray(t)))) for n, t in cp.items()})
for n, t in validated_gated8_checkpoint.items():
    delta = np.asarray(t) - np.asarray(load_original(n))
    if delta.ndim == 2:
        axis = 1 if ".0." in n else 0
        changed = np.flatnonzero(np.max(np.abs(delta), axis=axis) > 1e-7)
        print(n, "changed hidden indices:", changed.tolist(),
              "max change:", float(np.max(np.abs(delta))))
    else:
        print(n, "max change:", float(np.max(np.abs(delta))))

print("=== IDENTIFICATION NOTES CONTINUED ===")
print(notes_text[8500:12700])
print("=== EARLY NOTEBOOK HEADINGS ===")
headings = [x for x in notes_text.splitlines() if x.startswith("## ")]
print("\n".join(headings[:45]))
```
