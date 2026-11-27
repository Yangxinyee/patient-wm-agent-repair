The posterior directions have weak desired responses and noticeable cross-drug effects. I’ll test centered corrections locally across several dose multipliers and factual readings before submitting one. A useful candidate should produce consistent dose responses without a material increase in factual NLL.

```python
import time
import numpy as np
import torch

t71 = time.time()
def arr71(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

post_keys71 = list(post66_base_weights)
backup71 = {k: W0[k].clone() for k in post_keys71}
C71 = np.asarray(post66_rate_models[0.1], dtype=np.float32)

def set_post71(D=None, strength=0.0):
    for k, v in post66_base_weights.items():
        W0[k] = torch.as_tensor(v, dtype=torch.float32).clone()
    if D is not None and strength:
        dc = torch.as_tensor(np.asarray(D) * strength, dtype=torch.float32)
        W0["z_post.2.weight"][:32] += dc @ torch.tensor(C71[:-1].T)
        W0["z_post.2.bias"][:32] += dc @ torch.tensor(C71[-1])

rng71 = np.random.default_rng(710)
fp71 = rng71.choice(len(fidx59), size=min(320, len(fidx59)), replace=False)
fi71 = np.asarray(fidx59)[fp71]
bins71 = np.nan_to_num(np.asarray(bins59)[fp71], nan=0).astype(int)
mask71 = (bins71 > 0) & (bins71 < 17)
def factual_local71():
    o, h = r59_core(fi71)
    result = r59_features(o, h, fi71)
    p = arr71(result[2]).mean(axis=2)
    chosen = p[np.arange(len(fi71))[:, None], np.arange(4)[None, :], bins71]
    return -np.log(np.maximum(chosen[mask71], 1e-12)).mean()

drug_idx71 = {}
cf_u71 = {}
for d, drug in enumerate(["insulin", "norepinephrine"]):
    eligible = np.flatnonzero(rate66[:, d] > 0)
    drug_idx71[d] = rng71.choice(eligible, size=min(64, len(eligible)), replace=False)
    for multiplier in [0.0, 0.5, 2.0, 3.0]:
        cf_u71[d, multiplier] = rr35_scaled_action_encoding(
            drug_idx71[d], drug, multiplier)

diagnostics71 = []
try:
    nll_original71 = factual_local71()
    print("Original local NLL:", nll_original71, "readings:", int(mask71.sum()))
    for lam, strength in [(0.001, 0.0), (0.001, 0.3), (0.001, 1.0),
                          (0.001, 3.0), (0.0001, 0.3), (0.0001, 1.0)]:
        set_post71(D56_candidates[lam], strength)
        nll = factual_local71()
        row = {"lambda": lam, "strength": strength,
               "dNLL": float(nll - nll_original71), "slopes": {}}
        for d in [0, 1]:
            ii = drug_idx71[d]
            o, h = r59_core(ii)
            baseline = arr71(r59_features(o, h, ii)[3])
            for multiplier in [0.0, 0.5, 2.0, 3.0]:
                o, h = r59_core(ii, u=cf_u71[d, multiplier])
                change = arr71(r59_features(o, h, ii)[3]) - baseline
                dose_change = rate66[ii, d] * (multiplier - 1.0)
                slope = (dose_change[:, None] * change).sum(axis=0) / (
                    np.square(dose_change).sum() + 1e-12)
                row["slopes"][d, multiplier] = np.round(slope, 3).tolist()
        diagnostics71.append(row)
        print(row)
finally:
    W0.update(backup71)

print("Seconds:", round(time.time()-t71, 2), "Submitted edits:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 71: local centered-posterior frontier\n"
            "- Tested explicit dose-innovation carrier plus regularized posterior response directions.\n"
            "- Assessed exact local categorical NLL and 0/0.5/2/3 dose responses; diagnostics71 retained.\n"
            "- Scratch reconstruction weights restored; no submitted edits.\n"
            "- Python runs: 71; forward passes: 4.\n")
```