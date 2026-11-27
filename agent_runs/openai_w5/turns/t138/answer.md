The stored candidate matches the best fit, and paired coordinate differences remove its carrier baseline. I’ll test fresh doses, including zero and mixed-drug bundles, then install the encoder if those checks pass. The subsequent GPU factual test—not the local encoding test—will determine whether its cost is acceptable.

```python
qcheck123, pcheck123 = channel_dataset117(2500, 1000)
_, _, _, ucheck123 = enc_set117(qcheck123, pcheck123, enc_candidate118)
_, _, _, uorig123 = enc_set117(qcheck123, pcheck123, enc_orig117)
active123 = pcheck123.any(axis=1)

decoded123 = np.column_stack([
    ucheck123[:, R117[0]] - ucheck123[:, R117[1]],
    ucheck123[:, R117[2]] - ucheck123[:, R117[3]],
]) * rate_scale105 / (2 * enc_delta117)
decoded123[~active123] = 0
err123 = decoded123 - qcheck123
rate_rmse123 = np.sqrt(np.mean(err123[active123]**2, axis=0))
ordinary_rmse123 = np.sqrt(np.mean(
    (ucheck123[active123][:, K117] - uorig123[active123][:, K117])**2
))

print("fresh rate RMSE I/NE:", rate_rmse123)
print("fresh maximum rate errors:", np.max(np.abs(err123), axis=0))
print("ordinary-feature RMSE:", ordinary_rmse123)
for label, mask in [
    ("insulin-only", pcheck123[:, 0] & ~pcheck123[:, 1]),
    ("NE-only", ~pcheck123[:, 0] & pcheck123[:, 1]),
    ("both", pcheck123.all(axis=1)),
    ("present-but-zero", active123 & (qcheck123.sum(axis=1) == 0)),
    ("empty", ~active123),
]:
    if mask.any():
        print(label, "n", int(mask.sum()),
              "rate RMSE", np.sqrt(np.mean(err123[mask]**2, axis=0)))

snapshot123 = {n: load(n) for n in edited()}
assert np.all(rate_rmse123 < np.array([0.015, 0.005]))
assert ordinary_rmse123 < 0.004
for n, tensor in enc_candidate118.items():
    save_edit(n, tensor)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 123: encoder candidate installed for factual validation\n"
        f"- Independent rate RMSE I/NE: {rate_rmse123.tolist()}.\n"
        f"- Ordinary action-feature RMSE: {float(ordinary_rmse123):.7f}.\n"
        "- Installed the best normalized analytic encoder; no causal readout coupling yet.\n"
        "- Existing coordinate/head ablations remain. GPU factual validation is required.\n"
    )
print("current edits:", edited())
print("next factual stays:", int(factual9[idcol122].min()),
      "through", int(factual9[idcol122].max()))
```
