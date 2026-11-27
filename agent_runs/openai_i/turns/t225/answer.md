The gated features are mostly zero on ordinary bundles, unlike the leaky prelude decoders. That makes them a better candidate input, but the proposed large latent-token anchor could disturb the existing model. I’ll first build a zero-effect version and measure how well it preserves the original attention output. If preservation is poor before any drug correction, this transplant should be rejected.

```python
# Construct a candidate in memory only; do not install any edits.
H201 = array196(hidden196).astype(np.float64)
HF201 = np.column_stack([H201, np.ones(len(H201))])
feature_coef201 = np.linalg.lstsq(HF201, F195, rcond=1e-8)[0]
feature_pred201 = HF201 @ feature_coef201
print("Innovation feature reconstruction RMS:",
      np.sqrt(np.mean((feature_pred201-F195)**2, axis=0)))

private_post201 = {
    k: torch.as_tensor(array196(v).copy(), dtype=torch.float32)
    for k, v in base169.items()
}
for j, dim in enumerate(private_dims196):
    private_post201["z_post.2.weight"][int(dim)] = torch.tensor(
        feature_coef201[:-1, j], dtype=torch.float32)
    private_post201["z_post.2.bias"][int(dim)] = float(feature_coef201[-1, j])

# Recover the exact correspondence between exposed states and prelude tokens.
all_z201 = array196(state_cal161)[:, -32:].astype(np.float32)
z_lookup201 = {row.tobytes(): i for i, row in enumerate(all_z201)}
source201 = np.array([
    z_lookup201.get(row.astype(np.float32).tobytes(), -1)
    for row in array196(zofficial193)
])
print("State/token matches:", int((source201 >= 0).sum()), "/", len(source201))
assert (source201 >= 0).all(), "Need explicit token source indices before calibration."

Z201 = Zbase196.copy()
Z201[:, private_dims196] = feature_pred201
zin_bias_base201 = zin_newb196 - anchor_scale196 * anchor196

original_out201 = array196(load_original("core.prelude.0.attn.out.weight"))
target_att201 = array196(torch.nn.functional.linear(
    old_attention_features196,
    torch.as_tensor(original_out201, dtype=torch.float32)
)).reshape(-1, 192).astype(np.float64)
fit_att201 = np.asarray(maskatt196, dtype=bool)
assert fit_att201.shape == (len(target_att201),)
keep_att201 = np.setdiff1d(np.arange(192),
                         np.arange(private_head196*48, (private_head196+1)*48))

neutral_trials201 = {}
for anchor_strength in (1., 5., 20., 100.):
    zw = zin_newW196.copy()
    zb = zin_bias_base201 + anchor_strength * anchor196
    new_tokens = old_tokens196.clone()
    new_tokens[:, 6] = torch.as_tensor(
        (Z201 @ zw.T + zb)[source201], dtype=torch.float32)
    features = array196(attention_features196(
        new_tokens, W["core.prelude.0.attn.q.weight"],
        W["core.prelude.0.attn.kv.weight"]
    )).reshape(-1, 192).astype(np.float64)
    X = np.column_stack([features[:, keep_att201], np.ones(len(features))])
    xf, yf = X[fit_att201], target_att201[fit_att201]
    covariance = xf.T @ xf
    penalty = 1e-5 * np.trace(covariance) / len(covariance)
    regularizer = np.eye(len(covariance)) * penalty
    regularizer[-1, -1] = 0
    coef = np.linalg.solve(covariance + regularizer, xf.T @ yf)
    pred = X @ coef
    hold_error = pred[~fit_att201] - target_att201[~fit_att201]
    neutral_trials201[anchor_strength] = {
        "zinW": zw, "zinb": zb, "coef": coef,
        "tokens": new_tokens, "features": features,
        "hold_rms": float(np.sqrt(np.mean(hold_error**2)))
    }
    print("anchor", anchor_strength,
          "held attention RMS error", neutral_trials201[anchor_strength]["hold_rms"],
          "relative", float(np.sqrt(np.mean(hold_error**2)) /
                            np.sqrt(np.mean(target_att201[~fit_att201]**2))))

print("Original attention-output RMS:",
      float(np.sqrt(np.mean(target_att201[~fit_att201]**2))))
print("Norm1 constant-coordinate denominator:",
      float(np.sum(array196(W["core.prelude.0.norm1.bias"]) /
                   array196(W["core.prelude.0.norm1.weight"]))))
print("Active edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 201 — neutral private-channel calibration\n"
            "- Fitted the two gated innovations into freed posterior coordinates "
            "by closed-form least squares; no gradients or optimizer.\n"
            "- Compared attention-output preservation across anchor scales, "
            "with the donor head reserved and remaining outputs refitted by regression.\n"
            "- Candidates remain in memory only; factual GPU validation is required.\n")
```
