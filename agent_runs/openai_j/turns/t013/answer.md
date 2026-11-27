Insulin’s short-term effect remains negative in the sensor-based check, but its dynamic-panel estimates are unstable. Norepinephrine’s panel estimates are more stable and positive, yet disagree with the latent-state fit. I will examine the fit assumptions and estimate responses to unexpected dose variation across several horizons. Consistent response shapes would give us more defensible rollout targets.

```python
print("=== EXISTING FIT IMPLEMENTATION ===")
try:
    src = inspect.getsource(kalman_nll)
    print(src[:6500])
except Exception:
    print("source unavailable; constants:", kalman_nll.__code__.co_consts[:35])
    print("local names:", kalman_nll.__code__.co_varnames)

print("\n=== RETAINED ARRAY SHAPES ===")
for key in ("Y", "A", "F", "C", "E", "fold", "dose_resid", "outcome_resid",
            "readings", "rates", "initial_mean", "initial_noise", "controls", "presence"):
    obj = globals().get(key)
    print(key, type(obj).__name__, getattr(obj, "shape", None))
print("nuisance model:", nuisance_model())

# Build a complete, past-and-current measured-history feature matrix.
from sklearn.ensemble import HistGradientBoostingRegressor
hist_parts = []
hist_labels = []
scales = np.array([40, 40, 12, 12])
for lag in (0, 1, 2, 3):
    vals = np.roll(PF, lag, axis=1)
    seen_now = np.roll(np.isfinite(PO), lag, axis=1).astype(float)
    if lag:
        vals[:, :lag] = pmed
        seen_now[:, :lag] = 0
    hist_parts.extend([(vals - pmed) / scales, seen_now])
    hist_labels.extend([f"last_{lag}_{s}" for s in MS] +
                       [f"seen_{lag}_{s}" for s in MS])
for lag in (1, 2, 3):
    vals = np.roll(PA, lag, axis=1)
    vals[:, :lag] = 0
    hist_parts.append(vals)
    hist_labels.extend([f"dose_{lag}_{s}" for s in DS])
hist_parts.append(np.broadcast_to(phours[None, :, None] / 24, (PN, PT, 1)))
hist_labels.append("hour")
HX = np.concatenate(hist_parts, axis=2).astype(np.float32)
HF = np.arange(PN) % 2
innov = np.zeros_like(PA)
policy_fit_summary = []
for dj in range(2):
    for held in (0, 1):
        trmask = (HF[:, None] != held) & (phours[None, :] >= 0) & (phours[None, :] < 24)
        temask = (HF[:, None] == held) & (phours[None, :] >= 0)
        hp = HistGradientBoostingRegressor(
            max_iter=180, max_leaf_nodes=25, learning_rate=0.08,
            l2_regularization=20, min_samples_leaf=150, random_state=182+dj
        )
        hp.fit(HX[trmask], PA[:, :, dj][trmask])
        innov[:, :, dj][temask] = PA[:, :, dj][temask] - hp.predict(HX[temask])
        policy_fit_summary.append((DS[dj], held,
                                   float(np.std(innov[:, :, dj][temask]))))
print("\n=== CROSS-FITTED DOSE INNOVATION SD ===")
print(policy_fit_summary)

# Local projections: residualise future readings against pre-dose history.
# These include subsequent policy feedback; they are not yet controlled-dose kernels.
lp_rows = []
for dj, sj in enumerate((0, 2)):
    for horizon in (1, 2, 4, 8):
        ts = np.arange(1, PT - horizon)
        out = PO[:, ts + horizon, sj]
        valid = np.isfinite(out)
        er = innov[:, ts, dj][valid]
        resid_y = np.zeros(valid.sum())
        si, ti = np.where(valid)
        fold_obs = HF[si]
        Xlp = HX[:, ts][valid]
        ylp = out[valid]
        for held in (0, 1):
            trm, tem = fold_obs != held, fold_obs == held
            hm = HistGradientBoostingRegressor(
                max_iter=150, max_leaf_nodes=25, learning_rate=0.08,
                l2_regularization=20, min_samples_leaf=150,
                random_state=260+dj+horizon
            )
            hm.fit(Xlp[trm], ylp[trm])
            resid_y[tem] = ylp[tem] - hm.predict(Xlp[tem])
        est = np.dot(er, resid_y) / np.dot(er, er)
        gs = np.bincount(si, weights=er * (resid_y - est * er), minlength=PN)
        ses = np.linalg.norm(gs) / np.dot(er, er)
        lp_rows.append(dict(drug=DS[dj], horizon=horizon, n=len(si),
                            response=est, se=ses))
lp_table = pd.DataFrame(lp_rows)
print("\n=== DOSE-INNOVATION LOCAL PROJECTIONS ===")
print(lp_table.round(5).to_string(index=False))
lp_table.to_csv("/work/local_projections.csv", index=False)

with open("/work/notes.md", "a") as f:
    f.write("\n## Dose-innovation local projections\n")
    f.write("- Cross-fitted policy uses current/past sensor information and past doses; no future features.\n")
    f.write("- These responses include subsequent clinician-policy feedback, unlike fixed-dose evaluation.\n")
    f.write(lp_table.to_string(index=False) + "\n")
    f.write("- Dynamic-panel insulin estimates unstable; not suitable as repair targets.\n")
    f.write("- Python runs: 10. Forward passes: 3. No edits yet.\n")
```
