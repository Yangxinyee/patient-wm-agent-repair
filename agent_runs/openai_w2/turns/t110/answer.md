The diagnostic stopped because I addressed the normalization table with the wrong key layout; no weights changed. I’ll inspect that layout and summarize the corrected errors by event type without assuming its structure. Concentrated errors would support a parsing issue; widespread errors would refute that explanation.

```python
print("=== NORMALIZATION TABLE LAYOUT ===")
print("numeric_specs keys:", list(numeric_specs)[:12])
for key95 in list(numeric_specs)[:3]:
    print(repr(key95), repr(numeric_specs[key95])[:1200])
print("code_ids:", code_ids)

diagnostics95 = []
for j95, idx95 in enumerate(rows91):
    mr95 = sm_repair.iloc[idx95]
    rr95 = raw_groups94.get(
        (int(mr95["stay_id"]), round(float(mr95["hour"]), 4)), [])
    events95 = tuple(sorted(r95["event"] for r95 in rr95))
    diagnostics95.append({
        "j": j95, "stay_id": int(mr95["stay_id"]), "hour": float(mr95["hour"]),
        "has_action": bool(mask91[j95]),
        "events": ",".join(events95) or "END_only",
        "error": float(error_correct94[j95])
    })
diag95 = pd.DataFrame(diagnostics95)
print("\n=== CORRECTED ERROR QUANTILES ===")
print(diag95.groupby("has_action")["error"]
      .quantile([0, .25, .5, .75, .9, .95, .99, 1]).to_string())
print("\n=== ERROR BY EVENT COMBINATION ===")
print(diag95[~diag95.has_action].groupby("events")["error"]
      .agg(["count", "mean", "max"]).sort_values("mean", ascending=False)
      .head(20).to_string())
print("\n=== LARGEST REMAINING ERRORS ===")
print(diag95.nlargest(12, "error").to_string(index=False))
with open("/work/notes.md", "a") as f95:
    f95.write("\n## Runs 93–95: empty-action convention recovered\n"
              "Saved forward states support the encoded empty action, not zero: "
              "no-action posterior RMSE falls from 0.13769 to 0.01344; "
              "drug bundles remain about 4e-5. Run 94's diagnostic stopped on "
              "normalization key layout, without editing weights.\n")
print("Current edits:", edited())
```
