The two innovations can be represented exactly in the freed posterior coordinates, and all token rows matched their exposed states. The preservation test stopped because the attention helper returns a tuple, not a single tensor. I’ll select its feature tensor and complete the test. Low held-stay reconstruction error would justify testing the zero-effect transplant on the GPU.

```python
neutral_trials201 = {}
for anchor_strength in (1., 5., 20., 100.):
    zw = zin_newW196.copy()
    zb = zin_bias_base201 + anchor_strength * anchor196
    new_tokens = old_tokens196.clone()
    new_tokens[:, 6] = torch.as_tensor(
        (Z201 @ zw.T + zb)[source201], dtype=torch.float32)

    raw_attention = attention_features196(
        new_tokens, W["core.prelude.0.attn.q.weight"],
        W["core.prelude.0.attn.kv.weight"])
    if isinstance(raw_attention, tuple):
        attention_values = raw_attention[0]
    else:
        attention_values = raw_attention
    features = array196(attention_values).reshape(-1, 192).astype(np.float64)

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

gamma202 = array196(W["core.prelude.0.norm1.weight"])
beta202 = array196(W["core.prelude.0.norm1.bias"])
print("Original attention-output RMS:",
      float(np.sqrt(np.mean(target_att201[~fit_att201]**2))))
print("Norm1 constant-coordinate denominator:", float(np.sum(beta202/gamma202)))
print("Norm1 gamma range:", float(gamma202.min()), float(gamma202.max()))
print("Active edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 201–202 — neutral private-channel calibration\n"
            "- The gated innovations have exact linear reconstruction from modified "
            "posterior hidden features (RMS below 1e-13).\n"
            "- Matched all 3,935 prelude-token rows to official state rows.\n"
            "- Corrected tuple-return handling in the attention helper and tested "
            "held attention-output reconstruction with a reserved donor head.\n"
            "- No edits installed; reconstruction is not factual certification.\n")
```
