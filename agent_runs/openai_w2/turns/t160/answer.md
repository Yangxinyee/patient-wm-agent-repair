The stronger edit raised factual NLL by 0.00585, exceeding the limit on this cohort, so it is not acceptable as tested. I’ll restore the validated edit and inspect lower-cost decoders and factual prediction shifts. A decoder with better held-out dose-response-to-factual-error tradeoff would justify another candidate; otherwise I’ll reduce the gains.

```python
save_edit("core.b_in.weight", fallback_recurrent139)
print("Restored validated gains [-5,+8].")

print("=== DECODER REPORTS ===")
if isinstance(prelude_reports112, pd.DataFrame):
    print(prelude_reports112.to_string(index=False))
else:
    print(repr(prelude_reports112)[:3500])
for name140 in ["fits123", "prelude_fits112"]:
    obj140 = globals().get(name140)
    if isinstance(obj140, dict):
        for key140, value140 in obj140.items():
            if isinstance(value140, dict) and "report" in value140:
                print(name140, key140, value140["report"])
            elif isinstance(value140, dict):
                for subkey140, subval140 in value140.items():
                    if isinstance(subval140, dict) and "report" in subval140:
                        print(name140, key140, subkey140, subval140["report"])

orig140 = pd.read_csv("/work/fwd_15/factual_original.csv")
edit140 = pd.read_csv("/work/fwd_15/factual_edited.csv")
joined140 = orig140.merge(edit140, on=["stay_id", "hour", "event", "observed"],
                         suffixes=("_original", "_edited"), validate="one_to_one")
joined140["mean_shift"] = joined140.predicted_mean_edited - joined140.predicted_mean_original
joined140["original_error"] = joined140.observed - joined140.predicted_mean_original
joined140["nll_change"] = joined140.nll_edited - joined140.nll_original
summary140 = []
for event140, part140 in joined140.groupby("event"):
    shifts140 = part140.mean_shift.to_numpy()
    errors140 = part140.original_error.to_numpy()
    summary140.append(dict(event=event140, n=len(part140),
                           mean_shift=shifts140.mean(),
                           rms_shift=np.sqrt(np.mean(shifts140**2)),
                           error_shift_cov=np.mean(errors140*shifts140),
                           nll_change=part140.nll_change.mean()))
print("=== STRONG CANDIDATE FACTUAL SHIFTS ===")
print(pd.DataFrame(summary140).to_string(index=False))
print("=== FEATURE STRUCTURES ===")
for name140 in ["prelude_features111", "prelude_cf_hidden111",
                "prelude_cf_E111", "use128", "Slocal128", "counter112"]:
    obj140 = globals().get(name140)
    if isinstance(obj140, dict):
        print(name140, {str(k): np.shape(v) for k,v in obj140.items()})
    elif obj140 is not None:
        print(name140, np.shape(obj140))
with open("/work/notes.md", "a") as f140:
    f140.write("\n## Forward 15 / run 140: stronger gains rejected on first cohort\n"
               "Gains [-10,+12] factual NLL 1.58059 vs 1.57474 (+0.00585); "
               "exceeds budget. Restored exact fallback_recurrent139 [-5,+8]. "
               "Inspecting alternative decoder efficiency and factual shifts before "
               "choosing a revised candidate. No rollout spent on failed candidate.\n")
```
