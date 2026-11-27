The trial predicts shifts of −13.00/+5.34 for glucose and +5.75/−2.79 for MAP, improving all four over the original. It still undershoots the record-based estimates. I’ll preserve it as a validated fallback and review whether the decoder can improve dose sensitivity without sacrificing factual fit.

```python
validated_B225 = np.asarray(load("core.b_in.weight"), dtype=np.float32).copy()
validated_summary225 = dict(summary224)
roll225 = pd.read_csv("/work/fwd_38/dose_edit_edited.csv")
reach225 = roll225["reached"]
if reach225.dtype == object:
    reach225 = reach225.astype(str).str.lower().isin(["true","1"])
else:
    reach225 = reach225.astype(bool)
validated_rollout225 = roll225.loc[reach225].groupby("edit")["shift"].mean().to_dict()
print("VALIDATED FALLBACK:", hashlib.sha256(validated_B225.tobytes()).hexdigest()[:12])
print("ROLLOUT:", validated_rollout225)
print("DECOMPOSITION MAX ERROR:",
      np.max(np.abs(validated_B225 -
                    (np.asarray(load_original("core.b_in.weight")) +
                     1.20*delta_ins218 + 1.05*delta_ne218))))

notes225 = Path("/work/notes.md").read_text()
print("\nEARLY RECORD-EVIDENCE NOTEBOOK (first 6200 characters)")
print(notes225[:6200])

print("\nCONSTRAINED INSULIN FIT DIAGNOSTICS")
print(table206.to_string(index=False))
print("\nCONSTRAINT TARGETS")
for name225, val225 in targets206.items():
    print(name225, np.round(np.asarray(val225),6).tolist())
print("\nPROBE MEANS:", {k: (np.asarray(v).shape if np.asarray(v).size > 12 else v)
                         for k,v in probe_means206.items()})
print("CONSTRAINED FIT KEYS:", list(constrained_fits206.keys())[:12])

with open("/work/notes.md","a") as f225:
    f225.write("\n## Forward38 / Run225: reduced-NE trial validated and frozen\n")
    f225.write("Validated fallback validated_B225: only core.b_in.weight; gains 1.20 insulin / 1.05 NE.\n")
    f225.write("Four matched rollout shifts: "+json.dumps(validated_rollout225)+"\n")
    f225.write("Full tuning factual delta 0.00330373, clustered SE 0.000882254, one-sided 95% upper 0.00475503.\n")
    f225.write("Further changes require factual and rollout validation; no weights changed in this run.\n")
```
