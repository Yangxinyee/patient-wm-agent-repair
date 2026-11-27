The old factual features reproduce `X147` exactly, and the installed edit is unchanged. I can now add the independent training states to the decoder fit. I’ll compare candidates at matched norepinephrine probe sensitivity; lower factual-state residuals on both cohorts would support testing a safer edit, though only a forward pass can verify NLL.

```python
import pandas as pd
from scipy.linalg import solve

E202 = np.asarray(r198_prelude[1], dtype=np.float64)
X202 = np.column_stack([
    np.ones(E202.shape[0] * 4),
    (E202.reshape(-1, 192) - mu147) / sd147
])
Hnew202 = X202.T @ X202 / len(X202)
Hpool202 = (len(X147)*Hf171 + len(X202)*Hnew202) / (len(X147)+len(X202))

print("NEW FEATURES", E202.shape, X202.shape)
print("PROBE PARTS", [(k, np.asarray(d).shape, np.asarray(y).shape)
                      for k, d, y in zip(probe_keys171, Dparts147, Yparts147)])
assert np.allclose(D147, np.vstack(Dparts147))
assert np.allclose(Y147, np.vstack(Yparts147))

probe_labels202 = np.concatenate([
    np.full(len(d), j, dtype=np.int64)
    for j, d in enumerate(Dparts147)
])
ne_double202 = probe_labels202 == probe_keys171.index(("norepinephrine", 2.0))
ne_half202 = probe_labels202 == probe_keys171.index(("norepinephrine", 0.5))

Bcurrent202 = np.vstack([
    mu147 @ validated_W176,
    sd147[:, None] * validated_W176
])
print("CURRENT PREDICTION RECONSTRUCTION MAX ERROR",
      np.max(np.abs(X147 @ Bcurrent202 -
                    F147.reshape(-1,192) @ validated_W176)))
target_ne202 = float(np.mean((D147 @ Bcurrent202)[ne_double202, 1]))
print("MATCHED NE DOUBLE PROBE MEAN", target_ne202)

def rms202(a):
    return float(np.sqrt(np.mean(np.asarray(a)**2)))

print("CURRENT DECODER FACTUAL RMS: OLD / INDEPENDENT",
      [rms202((X147 @ Bcurrent202)[:,j]) for j in range(2)],
      [rms202((X202 @ Bcurrent202)[:,j]) for j in range(2)])

fits202, rows202 = {}, []
for pw202 in [0.003, 0.01, 0.03, 0.1, 0.3]:
    for ridge202 in [1e-7, 1e-6, 1e-5, 1e-4]:
        b202 = fit_zero_fact171(Hpool202, Hp171, Gp171, pw202, ridge202)
        raw_sens202 = float(np.mean((D147 @ b202)[ne_double202, 1]))
        if raw_sens202 <= 0:
            continue
        b202[:,1] *= target_ne202 / raw_sens202
        w202 = fold_decoder147(b202)
        fits202[(pw202,ridge202)] = (b202.copy(), w202.copy())
        p202 = D147 @ b202
        rows202.append({
            "probe_weight": pw202, "ridge": ridge202,
            "old_fact_NE_RMS": rms202((X147 @ b202)[:,1]),
            "new_fact_NE_RMS": rms202((X202 @ b202)[:,1]),
            "NE_half_probe": float(p202[ne_half202,1].mean()),
            "NE_double_probe": float(p202[ne_double202,1].mean()),
            "initial_max": float(np.max(np.abs(C147 @ b202))),
            "fold_max_error": float(np.max(np.abs(
                X202 @ b202 - E202.reshape(-1,192) @ w202)))
        })
diagnostics202 = pd.DataFrame(rows202)
print(diagnostics202.to_string(index=False, float_format=lambda x:f"{x:.6g}"))

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs200–202: independent decoder design validated\n"
            "Current edit still exactly validated_B176. F147/Efull134 reproduce "
            "X147 exactly under the saved standardization. The normalized second "
            "prelude return supplies independent training features. Comparing "
            "pooled zero-factual-target fits at matched current NE doubled-dose "
            "probe sensitivity; these residual diagnostics are not factual NLL. "
            "No weight changes.\n")
```
