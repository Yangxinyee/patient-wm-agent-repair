Unexpected dose variation predicts lower glucose after insulin and higher MAP after norepinephrine. Those responses still include later dose changes, so they are not fixed-dose effects. I will next check whether measurement error explains the conflicting norepinephrine estimates. Agreement between sensor-swapped estimates would support that explanation; substantial disagreement would weaken it.

```python
# Use only information recorded before the current hour.
HPAST = HX[:, :, 8:]
eiv_rows = []
eiv_cache = {}

for dj, (s1, s2) in enumerate(((0, 1), (2, 3))):
    ts = np.arange(1, PT - 1)
    valid = (np.isfinite(PO[:, ts, s1]) & np.isfinite(PO[:, ts, s2])
             & np.isfinite(PO[:, ts + 1, s1]))
    si, ti = np.where(valid)
    tt = ts[ti]
    feats = HPAST[si, tt]
    targets = np.column_stack([
        PO[si, tt, s1], PO[si, tt, s2],
        PA[si, tt, dj], PO[si, tt + 1, s1]
    ])
    residuals = np.zeros_like(targets)
    for col in range(4):
        for held in (0, 1):
            trm = HF[si] != held
            tem = ~trm
            hm = HistGradientBoostingRegressor(
                max_iter=180, max_leaf_nodes=20, learning_rate=0.075,
                l2_regularization=20, min_samples_leaf=100,
                early_stopping=False, random_state=380+col+dj
            )
            hm.fit(feats[trm], targets[trm, col])
            residuals[tem, col] = targets[tem, col] - hm.predict(feats[tem])
    eiv_cache[DS[dj]] = (si, tt, residuals)
    r1, r2, ra, ry = residuals.T
    dd = r1 - r2
    print("\n", DS[dj], "residual covariance [sensor1, sensor2, dose, future]:")
    print(np.round(residuals.T @ residuals / len(si), 5))
    print("noise-variance moments:",
          np.dot(dd, r1)/len(si), -np.dot(dd, r2)/len(si),
          "dose/noise-difference covariance:", np.dot(dd, ra)/len(si))
    for swap in (False, True):
        xobs, proxy = (r2, r1) if swap else (r1, r2)
        Xe = np.column_stack([xobs, ra])
        Ze = np.column_stack([proxy, ra])
        be, se_e, er_e = iv_2sls(Xe, ry, Ze, si)
        eiv_rows.append(dict(drug=DS[dj], swap=swap, n=len(si),
                             rho=be[0], rho_se=se_e[0],
                             beta=be[1], beta_se=se_e[1]))

eiv_table = pd.DataFrame(eiv_rows)
print("\n=== PAST-CONTROLLED, SENSOR-PROXY IV ===")
print(eiv_table.round(5).to_string(index=False))
print("Validity requires dose to be independent of the regressor sensor's measurement error.")
eiv_table.to_csv("/work/proxy_iv.csv", index=False)

print("\n=== SENSOR-DISAGREEMENT / DOSE LEAD-LAG ASSOCIATIONS ===")
lead_rows = []
for dj, (s1, s2) in enumerate(((0, 1), (2, 3))):
    for lead in (-2, -1, 0, 1, 2, 3):
        ts = np.arange(max(3, -lead), min(PT, PT-lead))
        valid = np.isfinite(PO[:, ts, s1]) & np.isfinite(PO[:, ts, s2])
        si, ti = np.where(valid)
        tt = ts[ti]
        dd = PO[si, tt, s1] - PO[si, tt, s2]
        aa = PA[si, tt+lead, dj]
        past = (PF[si, tt-1]-pmed) / scales
        Clead = np.column_stack([np.ones(len(si)), phours[tt]/24,
                                 past, past**2])
        dr = dd - Clead @ sla.lstsq(Clead, dd, lapack_driver="gelsy")[0]
        ar = aa - Clead @ sla.lstsq(Clead, aa, lapack_driver="gelsy")[0]
        lead_rows.append(dict(drug=DS[dj], dose_lead=lead,
                              slope=np.dot(dr, ar)/np.dot(dr, dr)))
print(pd.DataFrame(lead_rows).round(6).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Sensor-proxy dynamic check\n")
    f.write("- Instrument current sensor with its paired sensor, controlling only earlier history.\n")
    f.write("- Additional assumption: dose is independent of regressor sensor measurement noise; test sensor-swap stability.\n")
    f.write(eiv_table.to_string(index=False) + "\n")
    f.write("- Python runs: 11. Forward passes: 3. No edits yet.\n")
```
