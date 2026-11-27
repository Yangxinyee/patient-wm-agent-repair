The earlier head-only diagnostic failed: recovering even a small dose response caused large held-out factual perturbations. The probe changed raw memory slots much more than the normalized output. I’ll check whether usable dose information exists in those raw slots; low factual error with strong held-out probe response would justify a targeted core edit.

```python
from scipy.linalg import solve

with np.load("/work/fwd_3/states_original.npz") as zf:
    raw_ref = {k: zf[k].copy() for k in zf.files}
with np.load("/work/fwd_5/states_edited.npz") as zf:
    raw_probe = {k: zf[k].copy() for k in zf.files}
assert np.array_equal(raw_ref["stay_id"], raw_probe["stay_id"])
assert np.allclose(raw_ref["hour"], raw_probe["hour"])

xr = raw_ref["x"].astype(np.float64)
xp = raw_probe["x"].astype(np.float64)
hr = xr[:, 192:960].reshape(-1, 4, 192)
hp = xp[:, 192:960].reshape(-1, 4, 192)
print("STATE DIFFERENCES:")
for label, aa, bb in [
    ("out", xr[:, :192], xp[:, :192]),
    ("mean_h", hr.mean(1), hp.mean(1)),
    ("all_h", hr.reshape(-1, 768), hp.reshape(-1, 768)),
    ("posterior_z", xr[:, 960:], xp[:, 960:])]:
    print(label, "factual RMS", np.sqrt(np.mean(aa**2)),
          "probe delta RMS", np.sqrt(np.mean((bb-aa)**2)))

# Probe targets are decayed added exposures, not estimates of true outcomes.
probe_state_targets = np.zeros((len(xr), 2))
drug_rows = train[train.event.isin(["insulin", "norepinephrine"])]
for sid in np.unique(raw_ref["stay_id"]):
    indices = np.flatnonzero(raw_ref["stay_id"] == sid)
    times = raw_ref["hour"][indices]
    records = drug_rows[drug_rows.stay_id == sid]
    for j, (drug, increment, rho) in enumerate([
        ("insulin", 0.5, 0.9641), ("norepinephrine", 0.3, 0.8445)]):
        starts = records.loc[records.event == drug, "hour"].to_numpy()
        elapsed = times[:, None] - starts[None, :]
        weights = np.where(elapsed >= -1e-5,
                           rho**np.maximum(elapsed, 0), 0)
        probe_state_targets[indices, j] = increment * weights.sum(1)

unique_ids = np.unique(raw_ref["stay_id"])
rng = np.random.default_rng(711)
rng.shuffle(unique_ids)
fit_ids = unique_ids[:int(.7 * len(unique_ids))]
fit_mask = np.isin(raw_ref["stay_id"], fit_ids)
check_mask = ~fit_mask
raw_state_decoder_results = []
raw_state_decoder_fits = {}

for label, aa, bb in [
    ("out", xr[:, :192], xp[:, :192]),
    ("mean_h", hr.mean(1), hp.mean(1)),
    ("all_h", hr.reshape(-1, 768), hp.reshape(-1, 768)),
    ("posterior_z", xr[:, 960:], xp[:, 960:])]:
    mu = aa[fit_mask].mean(0)
    sd = np.maximum(aa[fit_mask].std(0), 1e-5)
    A = (aa - mu) / sd
    D = (bb - aa) / sd
    Af, Df, Tf = A[fit_mask], D[fit_mask], probe_state_targets[fit_mask]
    for strength in [1., 16., 256., 4096.]:
        covariance = Af.T @ Af + strength * Df.T @ Df
        covariance += np.eye(covariance.shape[0]) * (len(Af) * .01)
        coef = solve(covariance, strength * Df.T @ Tf, assume_a="pos")
        prediction = D[check_mask] @ coef
        factual = A[check_mask] @ coef
        gains = (prediction * probe_state_targets[check_mask]).sum(0) / (
            probe_state_targets[check_mask]**2).sum(0).clip(1e-12)
        row = {"features": label, "strength": strength,
               "insulin_gain": round(float(gains[0]), 4),
               "insulin_factual_rms": round(float(np.sqrt(np.mean(factual[:, 0]**2))), 4),
               "ne_gain": round(float(gains[1]), 4),
               "ne_factual_rms": round(float(np.sqrt(np.mean(factual[:, 1]**2))), 4)}
        raw_state_decoder_results.append(row)
        raw_state_decoder_fits[(label, strength)] = (coef, mu, sd)
print(pd.DataFrame(raw_state_decoder_results).to_string(index=False))
```
