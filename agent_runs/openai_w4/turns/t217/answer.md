The gates are essentially saturated on active bundles, so gate suppression does not explain most of the attenuation. I’ll recalibrate the raw decoder against known input-dose changes while anchoring its factual outputs. Near-unit held-out dose slopes with small factual-signal changes would support using the recalibrated channel.

```python
from scipy.linalg import solve

raw_base192 = np.asarray(raw87, dtype=np.float64)
rate192 = np.asarray(delta185, dtype=np.float64)
active192 = np.asarray(active_ne180, dtype=bool)
train192 = np.asarray(train180, dtype=bool)
test192 = np.asarray(test180, dtype=bool)

assert np.array_equal(ix180, indices102)
assert np.array_equal(np.flatnonzero(active192), active_pos187)
assert np.all(rate192[~active192] == 0)

raw_cf192 = {}
for factor in (0.5, 2.0):
    raw_cf192[factor] = raw_base192.copy()
    raw_cf192[factor][active_pos187, 384:576] = cf_actions187[factor]
    print("CF parity", factor,
          float(np.sqrt(np.mean(
              (raw_cf192[factor] -
               np.asarray(raw_cf87[("norepinephrine", factor)]))**2))))

mean192 = raw_base192[train192].mean(axis=0)
std192 = np.maximum(raw_base192[train192].std(axis=0), 1e-3)
A192 = np.column_stack([
    (raw_base192 - mean192) / std192,
    np.ones(len(raw_base192))
])
base_coef192 = np.asarray(coef_ne186, dtype=np.float64)
base_q192 = raw_base192 @ base_coef192[:576] + base_coef192[576]

D_parts192, residual_parts192 = [], []
for factor in (0.5, 2.0):
    difference = raw_cf192[factor] - raw_base192
    D = np.column_stack([difference / std192,
                         np.zeros(len(difference))])
    mask = train192 & active192
    D_parts192.append(D[mask])
    residual_parts192.append(
        (factor - 1) * rate192[mask] -
        difference[mask] @ base_coef192[:576]
    )

D192 = np.concatenate(D_parts192)
residual192 = np.concatenate(residual_parts192)
gram_anchor192 = A192[train192].T @ A192[train192] / train192.sum()
gram_delta192 = D192.T @ D192 / len(D192)
rhs_delta192 = D192.T @ residual192 / len(D192)
penalty192 = np.eye(577)
penalty192[-1, -1] = 1e-5

def dose_metrics192(coef):
    factual = gated_ne185(raw_base192, coef)
    mask = test192 & active192
    result = []
    for factor in (0.5, 2.0):
        change = gated_ne185(raw_cf192[factor], coef) - factual
        target = (factor - 1) * rate192
        slope = np.dot(change[mask], target[mask]) / np.dot(target[mask], target[mask])
        rmse = np.sqrt(np.mean((change[mask] - target[mask])**2))
        result.extend([float(slope), float(rmse)])
    return result

print("\nBASE [half slope, RMSE, double slope, RMSE]",
      np.round(dose_metrics192(base_coef192), 5))

calibrations192 = {}
for ridge in (0.001, 0.01):
    for strength in (1.0, 10.0, 100.0, 1000.0):
        correction = solve(
            gram_anchor192 + strength * gram_delta192 + ridge * penalty192,
            strength * rhs_delta192,
            assume_a="pos"
        )
        coef = base_coef192.copy()
        coef[:576] += correction[:576] / std192
        coef[576] += correction[576] - mean192 @ (correction[:576] / std192)
        factual_change = gated_ne185(raw_base192, coef) - gated_ne185(raw_base192, base_coef192)
        anchor_rmse = float(np.sqrt(np.mean(
            factual_change[test192 & active192]**2
        )))
        metrics = dose_metrics192(coef)
        calibrations192[(ridge, strength)] = {
            "coef": coef, "anchor_rmse": anchor_rmse, "metrics": metrics
        }
        print("ridge", ridge, "strength", strength,
              "held factual-signal RMSE", round(anchor_rmse, 5),
              "dose metrics", np.round(metrics, 5))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 192\n"
        "Run-191 gates were saturated on almost all held-out active bundles; "
        "raw NE decoder approximation is the main remaining transmission issue. "
        "Fitted closed-form decoder corrections, anchoring factual outputs and "
        "supervising imposed half/double physical-rate increments on training stays. "
        "Held-out transmission tests do not estimate drug effects. "
        "No saved weights changed.\n"
    )
```