This repeats the previous output, so it adds no new evidence; no weights have changed. I’ll run a compact alignment check and measure whether factual prediction errors track dose innovations. A consistent association would support a targeted correction, while a near-zero association would argue against it.

```python
print("=== RESIDUAL OBJECTS ===")
for label, store in (("lp", repair_lp), ("robust", repair_robust)):
    for key, val in store.items():
        print(label, key, type(val).__name__)
        if isinstance(val, dict):
            print({k: getattr(v, "shape", str(v)[:90]) for k, v in val.items()})
print("=== EXPOSURE INDEX ===")
print("s,j:", list(zip(repair_s[:5], repair_j[:5])))
for drug in repair_drugs:
    print(drug, repair_rates[drug].shape,
          repair_rates[drug][repair_s[:5], repair_j[:5]])
print(train[(train.stay_id == stay_ids[0]) & train.event.isin(repair_drugs)]
      [["hour", "event", "rate"]].head(8).to_string(index=False))

print("=== HEAD ALGEBRA ===")
print("context error:", (repair_context - repair_base - repair_gap).abs().max().item())
print("view error:", (repair_view - repair_context - lin(repair_prior, "z_head")).abs().max().item())
ff0 = lin(repair_withcode, "attr_heads.value_bin.0")
for ak, af in (("silu", TF.silu), ("gelu", TF.gelu), ("relu", TF.relu)):
    print("hidden difference", ak, ((af(ff0) - repair_phi)**2).mean().sqrt().item())
print("register numeric:", repair_num[:3].tolist())

repair_row_s = pd.Index(stay_ids).get_indexer(head_records.stay_id)
repair_row_h = head_records.hour.to_numpy()
repair_position = {int(j): k for k, j in enumerate(np.unique(repair_j))}
repair_effect_diagnostics = []
for event, drug in (("glucose_finger", "insulin"), ("map_arterial", "norepinephrine"),
                    ("glucose_lab", "insulin"), ("map_cuff", "norepinephrine")):
    key = ("glucose_finger", drug) if drug == "insulin" else ("map_arterial", drug)
    lp = repair_lp[key]
    rows = np.flatnonzero((head_records.event.to_numpy() == event) &
                          np.isin(repair_row_h, list(repair_position)))
    ix = np.array([repair_row_s[r] * len(repair_position) +
                   repair_position[int(repair_row_h[r])] for r in rows], dtype=int)
    good = lp["mask"][ix] & np.isfinite(lp["rd"][ix])
    rows, ix = rows[good], ix[good]
    rdv = lp["rd"][ix]
    err = head_records.value.to_numpy()[rows] - aligned_ref.predicted_mean.to_numpy()[rows]
    rdv = rdv - rdv.mean()
    err = err - err.mean()
    den = np.dot(rdv, rdv)
    slope = np.dot(rdv, err) / den if den else np.nan
    repair_effect_diagnostics.append((event, len(rows), np.std(rdv), slope))
print("=== INNOVATION SLOPE OF PREDICTION ERROR ===")
print(pd.DataFrame(repair_effect_diagnostics,
      columns=["event", "n", "innovation_sd", "error_slope"]).to_string(index=False))
rr = pd.read_csv("/work/fwd_5/dose_edit_original.csv")
print("Rollout columns:", list(rr.columns))
print(rr.head(2).to_string(index=False))
```