The recovered coefficients are the intended powers of two, so the probe can distinguish the numeric fields. I’ll now invert its predicted means and test register layouts, bin boundaries, and age units. Near-rounding-error agreement across all 4,510 readings would establish the layout; larger residuals would indicate another missing operation.

```python
from scipy.special import softmax

r90_meta = R85["meta"].reset_index(drop=True).copy()
r90_keys = ["stay_id", "hour", "event", "observed"]
assert r90_meta[r90_keys].equals(r87_probe[r90_keys].reset_index(drop=True))
r90_n = len(r90_meta)
r90_event = r90_meta.event.to_numpy()
r90_centres = np.array([
    [0.0] + list(VOCAB["value_bin_centres"][ev]) for ev in r90_event
], dtype=np.float64)

r90_slope = r87_probe_weights["attr_heads.value_bin.2.weight"][:, 0].numpy().astype(float)
r90_bias = r87_probe_weights["attr_heads.value_bin.2.bias"].numpy().astype(float)
r90_target = r87_probe.predicted_mean.to_numpy()
r90_lo, r90_hi = np.full(r90_n, -20.0), np.full(r90_n, 20.0)
for _ in range(55):
    r90_mid = (r90_lo + r90_hi) / 2
    r90_logits = (30.0 + r90_mid[:, None]) * r90_slope + r90_bias
    r90_mu = (softmax(r90_logits, axis=1) * r90_centres).sum(axis=1)
    r90_lo = np.where(r90_mu < r90_target, r90_mid, r90_lo)
    r90_hi = np.where(r90_mu >= r90_target, r90_mid, r90_hi)
r90_scalar = (r90_lo + r90_hi) / 2

r90_codes = {x["name"]: x["id"] for x in VOCAB["codes"]}
r90_codeid = np.array([r90_codes[ev] for ev in r90_event])
r90_has = r90_meta.has_last.to_numpy(dtype=float)
r90_z = r90_meta.last_z.to_numpy(dtype=float)
r90_age = r90_meta.age.to_numpy(dtype=float)
r90_dt = r90_meta.dt.to_numpy(dtype=float)
r90_numeric = {(x["code"], x["attribute"]): x for x in VOCAB["numeric"]}
r90_bins = {"cached": r90_meta.last_bin.to_numpy(dtype=int)}
for r90_side in ["left", "right"]:
    r90_lb = np.zeros(r90_n, dtype=int)
    for r90_ev in np.unique(r90_event):
        r90_ix = np.flatnonzero((r90_event == r90_ev) & (r90_has > 0))
        r90_spec = r90_numeric[(r90_ev, "value")]
        r90_last = r90_z[r90_ix] * r90_spec["iqr"] + r90_spec["median"]
        r90_lb[r90_ix] = 1 + np.searchsorted(
            r90_spec["fine_bin_edges"], np.round(r90_last, 5), side=r90_side)
    r90_bins[r90_side] = r90_lb

r90_codeW = load_original("embed.code.weight").numpy()
r90_valueW = load_original("embed.bins.value.weight").numpy()
r90_rateW = load_original("embed.bins.rate.weight").numpy()
r90_statusW = load_original("last_status.weight").numpy()
r90_nb = load_original("last_numeric.bias").numpy()
r90_age_options = {
    "state_seconds": np.log1p(np.maximum(r90_age, 0)) / 10,
    "state_hours": np.log1p(np.maximum(r90_age, 0) / 3600) / 10,
    "forecast_seconds": np.log1p(np.maximum(r90_dt, 1)) / 10,
    "forecast_hours": np.log1p(np.maximum(r90_dt, 1) / 3600) / 10,
}
r90_results = []
for r90_bn, r90_lb in r90_bins.items():
    for r90_sn, r90_si in [
        ("zero", np.zeros(r90_n, int)),
        ("one", np.ones(r90_n, int)),
        ("has_last", r90_has.astype(int)),
        ("omitted", None),
    ]:
        r90_fixed = r90_codeW[r90_codeid] + r90_valueW[r90_lb] + r90_rateW[0] + r90_nb
        if r90_si is not None:
            r90_fixed = r90_fixed + r90_statusW[r90_si]
        r90_resid = r90_scalar - r90_fixed @ r89_pw
        for r90_an, r90_af in r90_age_options.items():
            for r90_mask_age in [False, True]:
                r90_af_used = r90_af * r90_has if r90_mask_age else r90_af
                r90_X = np.column_stack([r90_has, r90_af_used, r90_z])
                r90_coef = np.linalg.lstsq(r90_X, r90_resid, rcond=None)[0]
                r90_err = r90_resid - r90_X @ r90_coef
                r90_results.append({
                    "bins": r90_bn, "status": r90_sn, "age": r90_an,
                    "mask_age": r90_mask_age,
                    "scalar_rmse": float(np.sqrt(np.mean(r90_err**2))),
                    "max_error": float(np.max(np.abs(r90_err))),
                    "coefficients_has_age_z": r90_coef.tolist(),
                    "missing_last_rmse": float(np.sqrt(np.mean(r90_err[r90_has == 0]**2))),
                })

r90_results.sort(key=lambda x: x["scalar_rmse"])
print("Best layouts:")
for r90_result in r90_results[:10]:
    print(r90_result)
print("Scalar range:", np.min(r90_scalar), np.max(r90_scalar))
print("Probe rows:", r90_n, "Current edits:", edited())
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Runs 88–90: register probe decoding\n"
            "Original weights restored. Probe numeric coefficients verified as "
            "[1,2,4,8,16,32,64,128]/128. Inverting saved probe means to test "
            "eight-field register layouts, age units, status, and bin boundaries.\n")
```