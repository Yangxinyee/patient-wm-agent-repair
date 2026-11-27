The paired-sensor check supports a negative insulin effect, but it cannot identify norepinephrine’s effect because its dose association is too weak. The zero placebo results were also mechanical: the controls included the placebo outcome. I will correct that check and compare dynamic-panel estimates; unstable estimates would rule them out as repair targets.

```python
import inspect

print("=== NUMERIC VOCAB ===")
print(json.dumps(VOCAB["numeric"])[:3500])
print("\n=== RETAINED STATISTICAL FUNCTIONS ===")
for key, obj in list(globals().items()):
    if callable(obj) and getattr(obj, "__module__", "") == "__main__" and key != "iv_ratio":
        try:
            print(key, inspect.signature(obj))
        except Exception:
            pass
print("\n=== LATENT FIT PARAMETERS ===")
for d in DS:
    print(d, kalman_fits[d].x)

def iv_2sls(X, y, Z, groups):
    zz = Z.T @ Z
    zx = Z.T @ X
    zy = Z.T @ y
    izz = np.linalg.pinv(zz, rcond=1e-11)
    bread = np.linalg.pinv(zx.T @ izz @ zx, rcond=1e-11)
    B = bread @ zx.T @ izz
    coef = B @ zy
    err = y - X @ coef
    gs = np.column_stack([
        np.bincount(groups, weights=Z[:, j] * err, minlength=PN)
        for j in range(Z.shape[1])
    ])
    cov = B @ (gs.T @ gs) @ B.T
    return coef, np.sqrt(np.maximum(np.diag(cov), 0)), err

# A non-mechanical placebo: only intercept and hour are controlled.
placebo_rows = []
for dj, (s1, s2) in enumerate(((0, 1), (2, 3))):
    ts = np.arange(2, PT)
    valid = (np.isfinite(PO[:, ts, s1]) & np.isfinite(PO[:, ts, s2])
             & np.isfinite(PO[:, ts - 1, s1]))
    si, ti = np.where(valid)
    tt = ts[ti]
    C0 = np.column_stack([np.ones(len(si)), phours[tt] / 24])
    b0, se0, fs0, rf0, sd0 = iv_ratio(
        PO[si, tt, s1] - PO[si, tt, s2],
        PA[si, tt, dj], PO[si, tt - 1, s1], si, C0
    )
    placebo_rows.append(dict(drug=DS[dj], n=len(si), beta=b0, se=se0,
                             dose_slope=fs0, past_outcome_slope=rf0))
print("\n=== CORRECTED PRE-TREATMENT PLACEBO ===")
print(pd.DataFrame(placebo_rows).round(5).to_string(index=False))

# Difference away stay-specific intercepts; instrument lagged dynamics.
gmm_rows = []
for dj, sensor_j in enumerate((0, 2)):
    for maxlag in (2, 3, 4):
        ts = np.arange(maxlag, PT - 1)
        valid = (np.isfinite(PO[:, ts + 1, sensor_j])
                 & np.isfinite(PO[:, ts, sensor_j])
                 & np.isfinite(PO[:, ts - 1, sensor_j]))
        for lag in range(2, maxlag + 1):
            valid &= np.isfinite(PO[:, ts - lag, sensor_j])
        si, ti = np.where(valid)
        tt = ts[ti]
        dy_next = PO[si, tt + 1, sensor_j] - PO[si, tt, sensor_j]
        dy_cur = PO[si, tt, sensor_j] - PO[si, tt - 1, sensor_j]
        da = PA[si, tt, dj] - PA[si, tt - 1, dj]
        Xg = np.column_stack([np.ones(len(si)), dy_cur, da])
        Zcols = [np.ones(len(si))]
        for lag in range(2, maxlag + 1):
            Zcols.append((PO[si, tt - lag, sensor_j] - pmed[sensor_j])
                         / (40 if dj == 0 else 12))
        for lag in range(1, maxlag + 1):
            Zcols.append(PA[si, tt - lag, dj])
        Zg = np.column_stack(Zcols)
        bg, sg, eg = iv_2sls(Xg, dy_next, Zg, si)
        gmm_rows.append(dict(drug=DS[dj], maxlag=maxlag, n=len(si),
                             rho=bg[1], rho_se=sg[1], beta=bg[2], beta_se=sg[2]))
gmm_table = pd.DataFrame(gmm_rows)
print("\n=== DIFFERENCED DYNAMIC-PANEL IV ===")
print(gmm_table.round(5).to_string(index=False))
gmm_table.to_csv("/work/dynamic_iv.csv", index=False)

print("\n=== HIGH-NOREPINEPHRINE RECORD EXAMPLE ===")
sid_ne = panel.groupby(level=0)["norepinephrine"].mean().nlargest(10).index[4]
print("stay:", sid_ne)
print(panel.loc[sid_ne].to_string())

with open("/work/notes.md", "a") as f:
    f.write("\n## Causal identification checks\n")
    f.write("- Paired-sensor IV suggests insulin lowers next glucose, approximately 2 per rate unit.\n")
    f.write("- Norepinephrine sensor-difference instrument is weak; do not use its IV ratios.\n")
    f.write("- Earlier zero placebos were mechanical because controls contained the placebo outcome; corrected here.\n")
    f.write(pd.DataFrame(placebo_rows).to_string(index=False) + "\n")
    f.write(gmm_table.to_string(index=False) + "\n")
    f.write("- Python runs: 9. Forward passes: 3. No edits yet.\n")
```
