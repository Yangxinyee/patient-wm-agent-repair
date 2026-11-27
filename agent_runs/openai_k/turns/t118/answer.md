Changing the gap scaling made the mismatch worse, so that is not the fix. I’ll next check query-type indices and the probability assigned to “no value.” This should distinguish another head-input error from a scoring-convention difference.

```python
print("=== VOCABULARY SEMANTICS ===")
print("top-level keys:", list(VOCAB))
for k111, v111 in VOCAB.items():
    if k111 not in ("normalization", "normalisation", "numeric", "bin_edges",
                    "value_bin_centres", "rate_bin_centres", "dose_bin_centres"):
        if len(str(v111)) < 1800:
            print(k111, v111)
print("event-kind map:", event_kind_ids_audit)
print("world kind map:", world_kq_audit)
print("centres, first four:", centres104[:4, :4])

print("\n=== NO-VALUE MASS ===")
mass111 = probe104[["event"]].copy()
mass111["p0"] = p110[:, 0].numpy()
print(mass111.groupby("event").p0.agg(["mean", "max"]).to_string())

type_results111 = []
for ti111 in range(15):
    for vc111 in (2, 6):
        for ac111 in ("gelu", "relu"):
            pp111, mm111, nn111 = assay_head110(0.1, vc111, 3, ac111, ti111)
            for ev111 in r49_sensors:
                mask111 = probe104.event.to_numpy() == ev111
                type_results111.append({
                    "event": ev111, "type": ti111, "value_col": vc111,
                    "activation": ac111,
                    "mean_mae": float(np.abs(mm111[mask111] -
                        probe104.predicted_mean.to_numpy()[mask111]).mean()),
                    "nll_mae": float(np.abs(nn111[mask111] -
                        probe104.nll.to_numpy()[mask111]).mean()),
                    "nll_bias": float((nn111[mask111] -
                        probe104.nll.to_numpy()[mask111]).mean())
                })
print("\n=== BEST TYPE CONVENTIONS BY SENSOR ===")
for ev111 in r49_sensors:
    print(ev111)
    for rr111 in sorted((r for r in type_results111 if r["event"] == ev111),
                        key=lambda r: r["nll_mae"])[:3]:
        print(rr111)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 111: query-type and no-value audit\n"
            "- Head gap scaling 1 versus .1 worsens reconstruction; rejected as the sole mismatch.\n"
            "- Tested all 15 query types and inspected no-value mass; no model edits.\n"
            "- Python runs: 111; forward passes: 4.\n")
```