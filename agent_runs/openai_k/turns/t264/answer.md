The prelude features show some separation: the best tested factual activity is about 19% of the average doubled-insulin contrast and 26% for norepinephrine. That warrants a cautious test, not confidence in a repair. I’ll check the value-head replay against official predictions and fit features in actual rate units; failed readout agreement would invalidate local output calibration.

```python
print("=== HEAD REPLAY AGAINST OFFICIAL FACTUAL MEANS ===")
r237_ref = pd.read_csv("/work/fwd_15/factual_original.csv")
r237_lh = r234_capture["r49_readouts"]
r237_rows = []
for r237_j, r237_event in enumerate(r49_sensors):
    r237_local = r235_result[r237_event]["M"].mean(axis=1)
    r237_next = r237_lh["hours49"] + r237_lh["dt_np49"]/3600
    for r237_k, r237_i in enumerate(r234_indices):
        r237_match = r237_ref[
            (r237_ref.stay_id == rr33_stays[r237_i]) &
            (np.abs(r237_ref.hour - r237_next[r237_k]) < 1e-5) &
            (r237_ref.event == r237_event)]
        if len(r237_match):
            r237_rows.append((int(rr33_stays[r237_i]), float(rr33_hours[r237_i]),
                              float(r237_next[r237_k]), r237_event,
                              float(r237_local[r237_k]),
                              float(r237_match.predicted_mean.iloc[0])))
r237_comparison = pd.DataFrame(r237_rows, columns=["stay", "state_hour", "reading_hour",
                                                  "event", "local", "official"])
print(r237_comparison.to_string(index=False))
if len(r237_comparison):
    print("mean absolute error by event",
          r237_comparison.assign(error=lambda x: abs(x.local-x.official))
          .groupby("event").error.mean().to_dict())

print("\n=== PHYSICAL DOSES AT REPLAYED BUNDLES ===")
r237_rate_table = tune[tune.event.isin(["insulin", "norepinephrine"])].pivot_table(
    index=["stay_id", "hour"], columns="event", values="rate", aggfunc="sum")
r237_rate_index = pd.MultiIndex.from_arrays(
    [rr33_stays[np.asarray(indices151)], rr33_hours[np.asarray(indices151)]],
    names=["stay_id", "hour"])
r237_doses = r237_rate_table.reindex(r237_rate_index).reindex(
    columns=["insulin", "norepinephrine"]).fillna(0).to_numpy()
print("dose means", r237_doses.mean(axis=0), "nonzero counts", (r237_doses>0).sum(axis=0))
print("selected-drug consistency RMS",
      np.sqrt(np.mean((r237_doses[np.arange(320), np.asarray(drug151, int)] -
                       np.asarray(dc151))**2)))
print("original E vs alpha-zero E RMS",
      float(torch.sqrt(torch.mean((base228["E"]-trials230[(0.0,"factual")]["E"])**2))))

print("\n=== PHYSICAL-RATE FEATURE FITS ===")
r237_features = {}
r237_A = trials230[(0.0, "factual")]["E"].numpy().reshape(-1,192).astype(np.float64)
for r237_j, r237_drug in enumerate(("insulin", "norepinephrine")):
    r237_D = (trials230[(0.0,r237_drug)]["E"] -
              trials230[(0.0,"factual")]["E"]).numpy().reshape(-1,192).astype(np.float64)
    r237_y = np.repeat(r237_doses[:,r237_j],4)
    for r237_lam in (1e-7,1e-6,1e-5,1e-4):
        r237_v = np.linalg.solve(
            (r237_A.T@r237_A + r237_D.T@r237_D)/len(r237_A) + r237_lam*np.eye(192),
            r237_D.T@r237_y/len(r237_A))
        r237_features[(r237_drug,r237_lam)] = r237_v
        r237_pred = r237_D@r237_v
        print(r237_drug, r237_lam,
              "factual RMS", round(float(np.sqrt(np.mean((r237_A@r237_v)**2))),4),
              "contrast slope", round(float(r237_pred@r237_y/(r237_y@r237_y)),4),
              "contrast RMSE", round(float(np.sqrt(np.mean((r237_pred-r237_y)**2))),4),
              "norm", round(float(np.linalg.norm(r237_v)),2))

print("\n=== ACTION ENCODING HELPERS ===")
print([(n, v.__code__.co_varnames[:v.__code__.co_argcount])
       for n,v in list(globals().items()) if isinstance(v,types.FunctionType)
       and any(s in n.lower() for s in ("action","embed","encode","atom"))])
with open("/work/notes.md", "a") as r237_f:
    r237_f.write("\n## Run 237: physical feature fits and head validation\n"
                 "- E has potentially useful factual-quiet directions, but held-out factual effects are not tested.\n"
                 "- Checking original local means against official factual predictions and fitting actual rate contrasts.\n"
                 "- No weights changed. Python runs: 237; forward passes: 17.\n")
```
