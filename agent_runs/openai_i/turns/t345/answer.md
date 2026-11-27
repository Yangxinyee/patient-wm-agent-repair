Data collection completed, but scoring stopped because the bin centres were still a Torch tensor. No compensation was applied. I’ll fix that type mismatch and reuse the collected features. The correction is worth a GPU test only if it improves held-out factual scoring while preserving the local dose responses.

```python
centres295 = array294(centres182).astype(np.float64)
base_means294 = np.sum(probabilities294(lbase294) * centres295[None], axis=-1)

def fit295(rank, ridge, mask):
    p = np.eye(192) - vh294[:rank].T @ vh294[:rank]
    xp = he294.astype(np.float64) @ p
    mu = xp[mask].mean(axis=0)
    sd = np.maximum(xp[mask].std(axis=0), 0.05)
    z = (xp - mu) / sd
    ym = target294[mask].mean(axis=0)
    coef = np.linalg.solve(
        z[mask].T @ z[mask] + ridge * mask.sum() * np.eye(192),
        z[mask].T @ (target294[mask] - ym)
    )
    raw = coef / sd[:, None]
    return p @ raw, ym - mu @ raw

def protected_shifts295(beta, bias):
    shifts = []
    for drug, scale, sel, hc, lc in protected_cases294:
        if scale not in (0.5, 2.0):
            continue
        own = events182.index(
            "glucose_finger" if drug == "insulin" else "map_arterial"
        )
        old_changed = np.sum(
            probabilities294(lc[sel]) * centres295[None], axis=-1
        )
        old_shift = float(np.mean(
            old_changed[:, own] - base_means294[sel, own]
        ))
        new_base = np.sum(
            probabilities294(lbase294[sel] + hbase294[sel] @ beta + bias)
            * centres295[None], axis=-1
        )
        new_changed = np.sum(
            probabilities294(lc[sel] + hc[sel] @ beta + bias)
            * centres295[None], axis=-1
        )
        new_shift = float(np.mean(new_changed[:, own] - new_base[:, own]))
        distortion = abs(new_shift - old_shift) / max(abs(old_shift), 0.25)
        shifts.append((drug, scale, old_shift, new_shift, distortion))
    return shifts

results295 = []
for rank in (4, 8, 12, 16, 24, 32):
    for ridge in (0.001, 0.01, 0.1, 1.0):
        beta, bias = fit295(rank, ridge, training294)
        new_logits = logits_e294 + he294 @ beta + bias
        residual = new_logits[held294] - logits_o294[held294]
        residual -= residual.mean(axis=1, keepdims=True)
        rms = float(np.sqrt(np.mean(residual**2)))
        distortion = max(s[-1] for s in protected_shifts295(beta, bias))
        dnll = nll294(new_logits, held294) - nll294(logits_o294, held294)
        results295.append((dnll, rms, distortion, rank, ridge))

eligible295 = [r for r in results295 if r[2] <= 0.15]
assert eligible295, sorted(results295)[:5]
best295 = min(
    eligible295, key=lambda r: r[0] if np.isfinite(r[0]) else r[1]
)
beta295, bias295 = fit295(
    best295[3], best295[4], np.ones(len(he294), dtype=bool)
)
candidate295 = {k: array294(v) for k, v in corrected269.items()}
candidate295["attr_heads.value_bin.2.weight"] = (
    array294(w2_294) + beta295.T
).astype(np.float32)
candidate295["attr_heads.value_bin.2.bias"] = (
    array294(b2_294) + bias295
).astype(np.float32)

print("Cases / labelled / held:",
      len(he294), int(valid294.sum()), int(held294.sum()))
print("Activation:", activation_name294, activation_errors294[activation_name294])
print("Held original / routed NLL:",
      nll294(logits_o294, held294), nll294(logits_e294, held294))
print("Protected direction energy:",
      {r: round(float(energy294[r-1]), 5) for r in (4, 8, 16, 32)})
print("Best (held dNLL, logit RMS, dose distortion, rank, ridge):", best295)
print("Top eligible:", sorted(eligible295)[:5])
print("Refitted local dose shifts:", protected_shifts295(beta295, bias295))

for name295, tensor295 in candidate295.items():
    save_edit(name295, tensor295)
np.savez("/work/protected_head295.npz", **candidate295)

tuning_ids295 = []
for chunk295 in pd.read_csv("/data/tuning.csv", usecols=["stay_id"], chunksize=50000):
    tuning_ids295 = list(dict.fromkeys(tuning_ids295 + chunk295.stay_id.tolist()))[:200]
    if len(tuning_ids295) == 200:
        break
print("TUNING COHORT A:", tuning_ids295[:100])
print("TUNING COHORT B:", tuning_ids295[100:200])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Python 295 — provisional protected-head correction\n"
        "- Run294 completed paired-state and dose-direction collection but stopped at a NumPy/Torch centre mismatch; no correction was installed then.\n"
        f"- Reused collected data: {len(he294)} cases, {valid294.sum()} recovered labels.\n"
        f"- Held original/routed local NLL: {nll294(logits_o294, held294):.6f} / {nll294(logits_e294, held294):.6f}.\n"
        f"- Selected (held dNLL, RMS, local dose distortion, protected rank, ridge): {best295}.\n"
        f"- Refitted local dose shifts: {protected_shifts295(beta295, bias295)}.\n"
        "- Installed protected_head295 provisionally. GPU factual validation on independent tuning cohorts and rollout validation are still required.\n"
        "- Correction is a closed-form regression, with no model gradients or optimiser over weights.\n"
    )
```