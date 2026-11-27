The paired action embeddings are available for stopping, halving, doubling and tripling each drug. I’ll measure their decoded dose changes with context held fixed, before testing downstream state changes. Near-linear, correctly signed contrasts would clear the carrier and shift attention to compensation and recurrence.

```python
r213_dec = np.asarray(restricted199, dtype=np.float64)
r213_x0 = latent206(U154)
r213_s0 = r213_x0 @ r213_dec
r213_signals = {}
r213_latents = {}

print("=== PHYSICAL RATE SUMMARY ===")
for d in range(2):
    a = np.asarray(rates187)[:, d]
    nz = a > 1e-8
    print("drug", d, "active", int(nz.sum()), "rate quantiles",
          np.quantile(a[nz], [0, .1, .5, .9, 1]))
print("factual signal RMS:", np.sqrt(np.mean(r213_s0**2, axis=0)))

print("\n=== FIXED-CONTEXT DOSE CONTRASTS ===")
print("drug scale n | true_mean decoded_mean slope relative_RMSE cross_mean")
for (d, s), u in sorted(scaled_u198.items()):
    x = latent206(u)
    signal = x @ r213_dec
    r213_latents[(d, s)] = x
    r213_signals[(d, s)] = signal
    mask = (np.asarray(rates187)[:, d] > 1e-8) & np.asarray(valid196, dtype=bool)
    if not mask.any():
        mask = np.asarray(rates187)[:, d] > 1e-8
    target = (s-1) * np.asarray(rates187)[mask, d]
    delta = (signal-r213_s0)[mask]
    slope = float(target @ delta[:, d] / max(target @ target, 1e-12))
    relerr = float(np.sqrt(np.mean((delta[:, d]-target)**2)) /
                   max(np.sqrt(np.mean(target**2)), 1e-12))
    print(d, s, int(mask.sum()),
          round(float(target.mean()), 5),
          round(float(delta[:, d].mean()), 5),
          round(slope, 4), round(relerr, 4),
          round(float(delta[:, 1-d].mean()), 5))

print("\n=== ROLLOUT BASELINE DRIFT ===")
r213_orig = pd.read_csv("/work/fwd_14/dose_edit_original.csv")
r213_join = r210_dose.merge(
    r213_orig, on=["stay_id", "hour", "edit"], suffixes=("_e", "_o")
)
r213_join["baseline_drift"] = r213_join["factual_e"]-r213_join["factual_o"]
print(r213_join[r213_join["reached_e"]].groupby("edit")["baseline_drift"].mean())

print("\n=== PRELUDE RETURN BYTECODE ===")
r213_buf = io.StringIO()
dis.dis(prelude155, file=r213_buf)
print(r213_buf.getvalue()[-1800:])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n\nPython run 213; forwards 14. Testing candidate206 normalized "
        "latent decoder against physical dose changes on paired action "
        "embeddings with fixed context, including stopping and tripling. "
        "Also checking rollout baseline drift. No weights changed.\n"
    )
```