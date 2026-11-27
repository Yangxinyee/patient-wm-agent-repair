The norepinephrine correction varies substantially even on factual states, explaining why amplification is costly. I’ll examine both drugs’ gate and dose-decoder distributions. If factual decoder values cluster near zero while the useful response comes from larger changes, a soft-threshold correction may preserve rollout effects with less factual disturbance; broad factual variation would argue against that approach.

```python
import numpy as np
import torch
from scipy.special import softmax

def r144_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

r144_X = r137_Xnp
r144_W0 = r144_np(r115_strength2["attr_heads.value_bin.0.weight"])
r144_b0 = r144_np(r115_strength2["attr_heads.value_bin.0.bias"])
r144_gw = (r144_W0[107] + r144_W0[109]) / 2
r144_gb = float((r144_b0[107] + r144_b0[109]) / 2)
r144_dw = (r144_W0[107] - r144_W0[109]) / 0.4
r144_db = float((r144_b0[107] - r144_b0[109]) / 0.4)
r144_G = r144_X @ r144_gw + r144_gb
r144_D = r144_X @ r144_dw + r144_db

print("=== INSULIN FEATURE RECONSTRUCTION ===")
print("G max error:", float(np.max(np.abs(r144_G - G104))))
print("D max error:", float(np.max(np.abs(r144_D - D104))))
print("theta104:", theta104)
print("tilt104:", np.asarray(tilt104).round(5).tolist())

r144_events = head_meta.event.to_numpy()
print("\n=== FACTUAL FEATURE QUANTILES ===")
for ev in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
    mask = r144_events == ev
    for name, arr in [
        ("insulin gate", r144_G),
        ("insulin decoder", r144_D),
        ("NE gate", r138_G),
        ("NE paired feature", r139_feature["pair"]),
    ]:
        z = np.asarray(arr)[:, mask].reshape(-1)
        print(ev, name, "mean/SD", np.round([z.mean(), z.std()], 4),
              "q01/10/50/90/99", np.round(np.quantile(z, [.01, .1, .5, .9, .99]), 4))

print("\n=== ACTIVE INSULIN FEATURE QUANTILES ===")
r144_active = np.asarray(active104, dtype=bool)
for name, arr in [("gate", r144_G), ("decoder", r144_D)]:
    z = arr[:, r144_active].reshape(-1)
    print(name, "n", int(r144_active.sum()),
          "mean/SD", np.round([z.mean(), z.std()], 4),
          "quantiles", np.round(np.quantile(z, [.01, .1, .5, .9, .99]), 4))

def r144_nll(logits):
    p = softmax(np.asarray(logits), axis=-1).mean(axis=0)
    return -np.log(np.maximum(p[np.arange(len(r138_y)), r138_y], 1e-30))

r144_current_loss = r144_nll(r139_expected_L)
r144_orig_loss = r144_nll(baseL104)
print("\n=== LOCAL FACTUAL LOSS BY EVENT ===")
for ev in sorted(set(r144_events)):
    mask = r144_events == ev
    print(ev, "n", int(mask.sum()),
          "original", float(r144_orig_loss[mask].mean()),
          "current delta", float((r144_current_loss-r144_orig_loss)[mask].mean()))

r144_formula = (
    np.asarray(baseL104) - np.asarray(removed104)
    + 2 * theta104 * np.asarray(R104)[..., None] * np.asarray(tilt104)
)
print("\nStrength-2 formula max error:",
      float(np.max(np.abs(r144_formula-r138_L0))))
```