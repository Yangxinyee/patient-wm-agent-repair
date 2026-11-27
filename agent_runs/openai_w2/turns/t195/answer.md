None of the existing decoder variants improves the trade-off substantially. I’ll fit an alternative that minimizes its signal on factual states while learning the supplied dose changes on dose probes, with the initial-state constraint retained. If it reduces factual variation at matched dose sensitivity, it merits held-out testing; otherwise I’ll keep the validated decoder.

```python
from scipy.linalg import solve

print("=== VERIFY EXISTING DESIGN MATRICES ===")
Xcheck171 = np.column_stack([
    np.ones(F147.shape[0] * 4),
    (F147.reshape(-1, 192).astype(np.float64) - mu147) / sd147
])
print("Factual design reconstruction:", np.max(np.abs(Xcheck171 - X147)))
assert np.allclose(Xcheck171, X147, atol=1e-7)

probe_keys171 = [
    ("insulin", 0.5), ("insulin", 2.0),
    ("norepinephrine", 0.5), ("norepinephrine", 2.0)
]
probe_masks171 = {}
for j171, key171 in enumerate(probe_keys171):
    target171 = np.asarray(residual_targets102[key171], dtype=np.float64)
    mask171 = np.any(np.abs(target171) > 1e-10, axis=1)
    probe_masks171[key171] = mask171
    diff171 = (
        np.asarray(prelude_cf_E111[key171], dtype=np.float64)
        - F147.astype(np.float64)
    )[mask171].reshape(-1, 192) / sd147
    design171 = np.column_stack([np.zeros(len(diff171)), diff171])
    response171 = np.repeat(target171[mask171], 4, axis=0)
    print(key171, design171.shape,
          "design error", np.max(np.abs(design171 - Dparts147[j171])),
          "target error", np.max(np.abs(response171 - Yparts147[j171])))
    assert np.allclose(design171, Dparts147[j171], atol=1e-6)
    assert np.allclose(response171, Yparts147[j171], atol=1e-7)

print("Factual Gram error:",
      np.max(np.abs(X147.T @ X147 / len(X147) - Hfact147)))
print("Probe Gram error:",
      np.max(np.abs(D147.T @ D147 / len(D147) - Hprobe147)))

Hf171 = X147.T @ X147 / len(X147)
Hp171 = D147.T @ D147 / len(D147)
Gp171 = D147.T @ Y147 / len(D147)
penalty171 = np.diag(np.r_[0.0, np.ones(192)])

def fit_zero_fact171(Hf, Hp, Gp, probe_weight, ridge):
    H = Hf + probe_weight * Hp + ridge * penalty171
    both = solve(H, np.column_stack([probe_weight * Gp, C147.T]),
                 assume_a="pos")
    B, HC = both[:, :2], both[:, 2:]
    B -= HC @ (np.linalg.pinv(C147 @ HC, rcond=1e-10) @ (C147 @ B))
    return B

zero_fact_decoders171 = {}
reports171 = []
reference171 = report_decoder147(validated_W151)
for pw171 in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
    for ridge171 in [1e-6, 1e-4, 1e-2]:
        Bzero171 = fit_zero_fact171(Hf171, Hp171, Gp171, pw171, ridge171)
        Wzero171 = fold_decoder147(Bzero171)
        Wzero171[:, 0] = validated_W151[:, 0]
        zero_fact_decoders171[(pw171, ridge171)] = Wzero171.copy()
        metrics171 = report_decoder147(Wzero171)
        gain171 = float(metrics171["ne_x2_gain"])
        scale171 = reference171["ne_x2_gain"] / gain171 if gain171 > 0 else np.nan
        reports171.append({
            "probe_weight": pw171, "ridge": ridge171,
            "ne_rms": float(metrics171["ne_rms"]),
            "ne_x2_gain": gain171,
            "ne_half_gain": float(metrics171["ne_half_gain"]),
            "matched_ne_rms": float(metrics171["ne_rms"] * scale171),
            "half_to_x2_gain": float(metrics171["ne_half_gain"] / gain171)
                              if gain171 > 0 else np.nan,
            "initial_max": float(metrics171["initial_max"])
        })

table171 = pd.DataFrame(reports171).sort_values("matched_ne_rms")
print("\n=== ZERO-FACTUAL-TARGET DECODER TRADE-OFFS ===")
print(table171.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
print("Validated matched NE RMS:", reference171["ne_rms"])

print("\n=== STATE FILE ALIGNMENT KEYS ===")
if "S" in globals():
    keys171 = list(S.files) if hasattr(S, "files") else list(S.keys())
    print([(k171, getattr(S[k171], "shape", None)) for k171 in keys171])
print("Weights unchanged:",
      np.array_equal(load("core.b_in.weight"), validated_B151))
```
