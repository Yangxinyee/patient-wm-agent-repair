The helpers expose both injection points, and the fitted coefficients can be converted into ordinary weight updates. I’ll check that conversion and recover the state-to-reading alignment for the local test. The conversion must reproduce the fitted residuals numerically; otherwise the proposed update would not implement what was fitted.

```python
print("=== ALIGNMENT ===")
print("probe_head110 columns:", list(probe_head110.columns))
print(probe_head110.head(5).to_string(index=False))
print("local_value_head dependencies:", local_value_head.__code__.co_names)
print("local_value_head constants:",
      [x for x in local_value_head.__code__.co_consts if isinstance(x, str)])
print("memory104 dependencies:", memory104.__code__.co_names)
print("memory104 constants:",
      [x for x in memory104.__code__.co_consts if isinstance(x, str)])

print("\n=== MEMORY AND INDEX VARIABLES ===")
for k in sorted(list(globals())):
    if any(s in k.lower() for s in ["memory", "mem104", "idx104", "indices104",
                                    "index110", "indices110", "idx110"]):
        v = globals()[k]
        print(k, describe115(v))

raw_fits117 = {}
for family117 in ["ffn", "E"]:
    fit117 = prelude_fits112[family117][0.3]
    w117 = fit117["B"][:-1] / fit117["sd"][:, None]
    b117 = fit117["B"][-1] - fit117["mu"] @ w117
    raw_fits117[family117] = (w117, b117)
    print("\n", family117, "raw weight norms",
          np.linalg.norm(w117, axis=0), "bias", b117)

gamma117 = Wrepair["core.prelude_norm.weight"].numpy().astype(np.float64)
bias117 = Wrepair["core.prelude_norm.bias"].numpy().astype(np.float64)
constant_vec117 = 1.0 / gamma117
constant117 = constant_vec117 @ bias117
w117, b117 = raw_fits117["E"]
E_linear_dose117 = w117 + np.outer(constant_vec117 / constant117, b117)
affine117 = prelude_E111 @ w117 + b117
linear117 = prelude_E111 @ E_linear_dose117
print("\nE layer-norm constant:", constant117)
print("Affine-to-linear conversion max error:",
      np.max(np.abs(linear117 - affine117)))
print("E factual residual RMS:", np.sqrt(np.mean(linear117 ** 2, axis=(0, 1))))
print("\nCurrent edits:", edited())
```
