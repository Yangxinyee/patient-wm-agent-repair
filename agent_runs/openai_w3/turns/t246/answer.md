Complete one-update responses have the expected signs, including a positive glucose shift for half-dose insulin. The failure therefore appears during longer rollout. The compensation was fitted before the carrier and gate changes, so I’ll refit it to the current features while penalizing changes to dose contrasts. Reduced baseline drift and a stronger half-dose response would support this explanation.

```python
import pandas as pd
import torch.nn.functional as F

print("=== GENERATED BASELINE BY HOUR ===")
print("rollout columns", list(r213_join.columns))
ins_labels220 = [x for x in r213_join["edit"].unique() if "ins" in str(x).lower()]
if ins_labels220:
    cols220 = [x for x in ["factual_e", "factual_o", "baseline_drift"]
               if x in r213_join.columns]
    print(r213_join[r213_join["edit"] == ins_labels220[0]]
          .groupby("hour")[cols220].mean().round(3).to_string())

w0_220 = tensor153(load_original("core.prelude.0.ffn.0.weight"))
b0_220 = tensor153(load_original("core.prelude.0.ffn.0.bias"))
w2_220 = tensor153(load_original("core.prelude.0.ffn.2.weight"))
b2_220 = tensor153(load_original("core.prelude.0.ffn.2.bias"))

def hidden220(after):
    return F.gelu(F.linear(layer_norm(after, "core.prelude.0.norm3"),
                           w0_220, b0_220))

orig_parts220 = prelude155(tokens_original155, overrides={})
orig_hidden220 = hidden220(orig_parts220[1])
orig_raw220 = orig_parts220[1] + F.linear(orig_hidden220, w2_220, b2_220)
current_hidden220 = hidden220(r214_base[1])
current_raw220 = r214_base[1] + F.linear(current_hidden220, w2_220, b2_220)

H220 = arr183(current_hidden220).reshape(-1, 768).astype(np.float64)
R220 = arr183(orig_raw220 - current_raw220).reshape(-1, 192).astype(np.float64)
tr220 = np.asarray(tr192, dtype=bool)
va220 = np.asarray(va192, dtype=bool)
assert len(tr220) == len(H220) and tr220.any() and va220.any()
hm220 = H220[tr220].mean(axis=0)
hs220 = np.maximum(H220[tr220].std(axis=0), .02)
X220 = np.column_stack([(H220 - hm220) / hs220, np.ones(len(H220))])
cov_base220 = X220[tr220].T @ X220[tr220] / tr220.sum()
rhs220 = X220[tr220].T @ R220[tr220] / tr220.sum()
cov_contrast220 = np.zeros((769, 769), dtype=np.float64)
contrast_features220 = {}
for key220 in sorted(r214_results):
    hscale220 = arr183(hidden220(r214_results[key220][1])).reshape(-1, 768)
    c220 = np.column_stack([
        (hscale220.astype(np.float64) - H220) / hs220,
        np.zeros(len(H220))
    ])
    cov_contrast220 += c220[tr220].T @ c220[tr220] / tr220.sum()
    contrast_features220[key220] = c220
cov_contrast220 /= len(contrast_features220)

ridge_diag220 = np.full(769, 1e-4)
ridge_diag220[-1] = 1e-8
refit220 = {}
refit220_scores = []
orig_Eflat220 = arr183(orig_parts220[0]).reshape(-1, 192)
norm_w220 = tensor153(load_original("core.prelude_norm.weight"))
norm_b220 = tensor153(load_original("core.prelude_norm.bias"))

print("\n=== CLOSED-FORM COMPENSATION REFITS ===")
for lam220 in [0., 1., 10., 100.]:
    theta220 = np.linalg.solve(
        cov_base220 + lam220 * cov_contrast220 + np.diag(ridge_diag220),
        rhs220
    )
    dw220 = (theta220[:768] / hs220[:, None]).T
    db220 = theta220[-1] - hm220 @ dw220.T
    fitted_raw220 = arr183(current_raw220).reshape(-1, 192) + H220 @ dw220.T + db220
    fitted_E220 = arr183(F.layer_norm(
        tensor153(fitted_raw220), (192,), norm_w220, norm_b220
    ))
    erms220 = float(np.sqrt(np.mean((fitted_E220[va220] - orig_Eflat220[va220]) ** 2)))
    rrms220 = float(np.sqrt(np.mean(
        (X220[va220] @ theta220 - R220[va220]) ** 2
    )))
    contrast_rms220 = float(np.sqrt(np.mean([
        np.mean((c220[va220] @ theta220) ** 2)
        for c220 in contrast_features220.values()
    ])))
    cand220 = {k: tensor153(v).clone() for k, v in candidate206.items()}
    cand220["core.prelude.0.ffn.2.weight"] = w2_220 + tensor153(dw220)
    cand220["core.prelude.0.ffn.2.bias"] = b2_220 + tensor153(db220)
    refit220[lam220] = cand220
    refit220_scores.append((lam220, erms220, rrms220, contrast_rms220))
    print("contrast penalty", lam220,
          "held-out E RMS", round(erms220, 8),
          "held-out raw RMS", round(rrms220, 5),
          "compensation contrast RMS", round(contrast_rms220, 5))

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 220; forwards 14. Complete one-update dose responses have the expected signs; asymmetry emerges during longer rollout. Refitted factual compensation to current restricted-carrier/action-only-gate features, with paired-dose contrast penalties and whole-stay holdout. Candidates retained but NOT installed. Scores: " + repr(refit220_scores) + "\n")
```