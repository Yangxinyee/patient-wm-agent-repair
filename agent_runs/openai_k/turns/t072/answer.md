The prior-to-head path has much stronger response directions, but the existing state poorly retains dose innovations. I’ll add explicitly dose-linear features to a local posterior copy and refit it to reproduce the original outputs. Small reconstruction error and near-unit held-out dose-response slopes would justify testing a dose-carrying state channel.

```python
from sklearn.linear_model import Ridge
t66 = time.time()

X66 = torch.as_tensor(rr41_aug_inputs, dtype=torch.float32)
W166 = W0["z_post.0.weight"].clone()
b166 = W0["z_post.0.bias"].clone()
W266 = W0["z_post.2.weight"].clone()
b266 = W0["z_post.2.bias"].clone()
F66old = act0(torch.nn.functional.linear(X66, W166, b166))
print("Original hidden agreement:",
      float((F66old - torch.as_tensor(rr41_H)).abs().max()))

importance66 = (
    F66old[:len(M59)].std(0)
    * W266.square().sum(0).sqrt()
).numpy()
reserve66 = np.argsort(importance66)[:4]
print("Reserved hidden rows:", reserve66.tolist())
decoder66 = np.asarray(rr39_action_decoder.coef_)[:2]
decoder_bias66 = np.asarray(rr39_action_decoder.intercept_)[:2]
for drug in range(2):
    for sign, row in zip([1., -1.], reserve66[2*drug:2*drug+2]):
        W166[row].zero_()
        W166[row, 384:576] = torch.as_tensor(sign * decoder66[drug])
        b166[row] = float(sign * decoder_bias66[drug])
F66new = act0(torch.nn.functional.linear(X66, W166, b166)).numpy()
Y66old = torch.nn.functional.linear(F66old, W266, b266).numpy()

index66 = np.asarray(rr41_orig_idx, dtype=int)
assert len(index66) == len(X66) == len(rr39_Y)
assert np.array_equal(index66[:len(M59)], np.arange(len(M59)))
train66 = ~held63[index66]

post66_clone = Ridge(alpha=0.001).fit(F66new[train66], Y66old[train66])
Y66clone = post66_clone.predict(F66new)
for label, mask in [
    ("held original bundles", (~held63) & (np.arange(len(M59)) >= 0)),
    ("held augmented bundles", ~train66)
]:
    if len(mask) == len(M59):
        err = Y66clone[:len(M59)][mask] - Y66old[:len(M59)][mask]
    else:
        err = Y66clone[mask] - Y66old[mask]
    print(label, "mean-coordinate RMSE", float(np.sqrt(np.mean(err[:, :32]**2))),
          "log-SD RMSE", float(np.sqrt(np.mean(err[:, 32:]**2))))

Z66 = np.column_stack([F66new, np.ones(len(F66new))]).astype(np.float64)
df66 = Z66[len(M59):] - Z66[index66[len(M59):]]
dr66 = np.asarray(rr39_Y)[:, :2] - np.asarray(rr39_rates)[index66]
assert np.max(np.abs(dr66[:len(M59)])) < 1e-6
rtrue66 = np.asarray(rr41_innov, dtype=np.float64)
rtrue66 = np.nan_to_num(rtrue66)
rate66 = np.asarray(rr39_rates)
print("Held true active innovation SD:",
      [float(rtrue66[held63 & (rate66[:, j] > 0), j].std())
       for j in range(2)])

post66_rate_models = {}
for contrast_penalty in [0.1, 1., 10., 100.]:
    fact_train = ~held63
    aug_train = train66[len(M59):]
    AX = np.concatenate([
        Z66[:len(M59)][fact_train],
        np.sqrt(contrast_penalty) * df66[aug_train]
    ])
    AY = np.concatenate([
        rtrue66[fact_train],
        np.sqrt(contrast_penalty) * dr66[len(M59):][aug_train]
    ])
    coefficient = np.linalg.solve(
        AX.T @ AX + 0.01 * np.eye(193), AX.T @ AY)
    post66_rate_models[contrast_penalty] = coefficient
    rp = Z66 @ coefficient
    delta = rp[len(M59):] - rp[index66[len(M59):]]
    active_rmse, absent_rms, slopes = [], [], []
    for j in range(2):
        active = held63 & (rate66[:, j] > 0)
        absent = held63 & (rate66[:, j] == 0)
        active_rmse.append(float(np.sqrt(np.mean(
            (rp[:len(M59)][active, j] - rtrue66[active, j])**2))))
        absent_rms.append(float(np.sqrt(np.mean(
            rp[:len(M59)][absent, j]**2))))
        use = (~train66[len(M59):]) & (np.abs(dr66[len(M59):, j]) > 1e-8)
        d = dr66[len(M59):][use, j]
        slopes.append(float(d @ delta[use, j] / (d @ d)))
    print("contrast penalty", contrast_penalty,
          "active innovation RMSE", np.round(active_rmse, 4).tolist(),
          "absent RMS", np.round(absent_rms, 4).tolist(),
          "held dose-change slopes", np.round(slopes, 4).tolist())

post66_base_weights = {
    "z_post.0.weight": W166,
    "z_post.0.bias": b166,
    "z_post.2.weight": torch.as_tensor(post66_clone.coef_, dtype=torch.float32),
    "z_post.2.bias": torch.as_tensor(post66_clone.intercept_, dtype=torch.float32)
}
print("Seconds:", round(time.time() - t66, 1), "Edits:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 66: explicit dose-linear posterior features\n"
            "- Prior-head sensitivities are substantially stronger than posterior-update sensitivities.\n"
            "- Existing state probes retain little dose innovation, motivating an explicit carrier.\n"
            "- Reserved four low-importance posterior hidden rows for GELU +/- pairs decoding the two rates.\n"
            "- Refit original posterior outputs and dose-centered readouts analytically; candidates retained, not submitted.\n"
            "- Python runs: 66; forward passes: 4; no tensors edited.\n")
```