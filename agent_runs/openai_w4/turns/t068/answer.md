The bundle check found a real discrepancy: many exported bundles contain more atoms than the raw rows, plausibly infusion-end markers. The posterior comparison also stopped because the context helper returns a tuple. I’ll inspect both before changing the reconstruction; end markers should explain the count differences exactly if that hypothesis is right.

```python
print("=== CONTEXT RETURN ===")
ctx_result59 = pre_context_repair(repair_bank, valid58[:4])
print(type(ctx_result59).__name__)
if isinstance(ctx_result59, tuple):
    for j59, value59 in enumerate(ctx_result59):
        print(j59, type(value59).__name__, getattr(value59, "shape", None))
print("context helper locals:", pre_context_repair.__code__.co_varnames)

print("=== BUNDLE COUNT DIFFERENCES ===")
difference59 = repair_bank["n_atoms"] - counts_e58 - counts_u58
print("raw atoms", len(repair_records), "exported atoms", int(repair_bank["n_atoms"].sum()))
print(pd.Series(difference59).value_counts().sort_index().to_string())
bad59 = np.flatnonzero(difference59 != 0)
for b59 in bad59[:10]:
    sid59 = int(repair_bank["stay_id"][b59])
    minute59 = int(np.rint(repair_bank["hour"][b59] * 60))
    same59 = repair_records[
        (repair_records["stay_id"] == sid59) &
        (repair_records["bundle_index"] == b59)]
    ending59 = repair_records[
        (repair_records["stay_id"] == sid59) &
        repair_records["event"].isin(["insulin", "norepinephrine"]) &
        (np.rint(repair_records["end_hour"].fillna(-999).to_numpy() * 60) == minute59)]
    print("state", b59, sid59, minute59, "n", int(repair_bank["n_atoms"][b59]),
          "flags", bool(repair_bank["has_insulin"][b59]),
          bool(repair_bank["has_norepinephrine"][b59]),
          "rows", same59[["event", "rate", "dose"]].to_dict("records"),
          "ending", ending59[["event", "rate", "dose"]].to_dict("records"))
print("=== DOSE VARIATION ===")
for drug59 in ("insulin", "norepinephrine"):
    rr59 = dml_residuals[drug59]
    print(drug59, "dose quantiles",
          np.quantile(rr59["dose"], [0, .25, .5, .75, .9, .99, 1]),
          "innovation quantiles",
          np.quantile(rr59["rd"], [0, .01, .1, .5, .9, .99, 1]))
    print("nuisance estimator pairs",
          [[type(est59).__name__ for est59 in pair59]
           for pair59 in dml_models[drug59]])
print("=== HEAD INNOVATION PROBE RESULTS ===")
for key59, val59 in head_innovation_probes.items():
    if isinstance(val59, dict):
        print(key59, {k: v for k, v in val59.items()
                      if np.isscalar(v) or isinstance(v, str)})
    else:
        print(key59, type(val59).__name__)
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 59\nRun-58 count check found 1,392 mismatched bundles; posterior test stopped because pre_context returns a tuple. Testing whether infusion-end atoms explain all count differences. No weights edited.\n")
```