The feed-forward correction reduced held-out slot error only from 0.77 to 0.60, so I won’t install it. The large routing anchor appears to disturb too much of the original computation. I’ll test whether the latent token can be selected with little or no anchor. Reliable held-out selection would let the other attention heads remain much closer to their originals.

```python
from sklearn.metrics import roc_auc_score

neutral221 = {k: np.array(v, copy=True) for k, v in neutral209.items()}
z221 = F.linear(
    F.gelu(F.linear(input196, t220(neutral221["z_post.0.weight"]),
                    t220(neutral221["z_post.0.bias"]))),
    t220(neutral221["z_post.2.weight"]),
    t220(neutral221["z_post.2.bias"]))[:, :32]
base_token221 = F.linear(z221, t220(neutral221["z_in.weight"]),
                        t220(neutral221["z_in.bias"]))
base_tokens221 = old_tokens196.clone()
base_tokens221[:, 6] = base_token221[source201]
fit_bundle221 = np.asarray(train196)[source201]
anchor_unit221 = np.asarray(anchor196, dtype=np.float64)
anchor_unit221 /= np.linalg.norm(anchor_unit221)
radius221 = float(torch.median(torch.linalg.vector_norm(base_token221, dim=1)))
gamma221 = orig220["core.prelude.0.norm1.weight"].numpy().astype(np.float64)
beta221 = orig220["core.prelude.0.norm1.bias"].numpy().astype(np.float64)
constant_sum221 = float(np.sum(beta221 / gamma221))
print("Original latent-token median norm:", radius221)
print("Neutral token RMS vs original:",
      rms220((base_tokens221[:, 6] - old_tokens196[:, 6]).numpy()))
print("Exact layer-norm constant:", constant_sum221)
print("Private coordinates:", private_indices209)
print("Private-coordinate percentiles:")
for j221 in np.asarray(private_indices209).reshape(-1):
    print(int(j221), np.percentile(z221[:, int(j221)].numpy(),
                                  [0, 1, 25, 50, 75, 99, 100]))

routes221 = []
for alpha221 in (0.0, 0.02, 0.1, 0.3, 1.0):
    tok221 = base_tokens221.clone()
    tok221[:, 6] += t220(alpha221 * radius221 * anchor_unit221)
    xn221 = F.layer_norm(tok221, (192,),
                         orig220["core.prelude.0.norm1.weight"],
                         orig220["core.prelude.0.norm1.bias"]).numpy().astype(np.float64)
    lat221 = xn221[fit_bundle221, 6]
    oth221 = xn221[fit_bundle221, :6].reshape(-1, 192)
    ml221, mo221 = lat221.mean(0), oth221.mean(0)
    dl221, do221 = lat221 - ml221, oth221 - mo221
    cov221 = 0.5 * (dl221.T @ dl221 / len(dl221) +
                   do221.T @ do221 / len(do221))
    ridge221 = 0.01 * np.trace(cov221) / 192
    gate221 = sla.solve(cov221 + ridge221 * np.eye(192), ml221 - mo221,
                        assume_a="pos", check_finite=False)
    gate221 /= float((ml221 - mo221) @ gate221)
    scores221 = xn221 @ gate221
    margin221 = scores221[:, 6] - scores221[:, :6].max(1)
    heldmargin221 = margin221[~fit_bundle221]
    probs221 = []
    for strength221 in (5., 10., 30.):
        p221 = torch.softmax(t220(strength221 * scores221), dim=1)[:, 6].numpy()
        probs221.append((strength221, float(p221[~fit_bundle221].mean()),
                         float(np.quantile(p221[~fit_bundle221], 0.01))))
    print("Anchor relative norm", alpha221,
          "held margin percentiles",
          np.round(np.percentile(heldmargin221, [0, 1, 10, 50, 99]), 3),
          "P(select latent):", probs221)
    routes221.append(dict(alpha=alpha221, gate=gate221, tokens=tok221,
                          xn=xn221, scores=scores221, probs=probs221,
                          heldmargin=heldmargin221))

print("Installed edits unchanged:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 221 — reject large-anchor compensation; test small-anchor selection\n"
            "- Large-anchor FFN compensation held RMS remains 0.600; not installed.\n"
            "- Tested closed-form latent-token selectors on stay-held bundles with progressively smaller anchors.\n"
            "- Neutral209 remains installed, with zero treatment-response columns.\n")
```