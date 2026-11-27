The inspection stopped on an object-type error before changing weights. The selected fit and its normalization statistics are available, so I’ll reconstruct and verify the mapping, then install the trial only if those checks pass. The hypothesis is that factual disturbance will fall while dose responses remain stable; the next factual and rollout tests can confirm or refute that.

```python
# All reconstruction checks occur before save_edit.
key210 = (("preserve_half", "preserve_cross"), (0.003, 1e-6))
coef210 = np.asarray(constrained_fits206[key210], dtype=np.float64)
raw_E210 = np.asarray(Efull134, dtype=np.float64).reshape(-1, 192)
design210 = np.asarray(X147, dtype=np.float64)
assert design210.shape == (len(raw_E210), 193)
standardization_error210 = np.max(np.abs(
    design210[:, :192] - (raw_E210 - mu147) / sd147))
assert standardization_error210 < 2e-4, standardization_error210
assert np.max(np.abs(design210[:, 192] - 1)) < 1e-6

# LayerNorm's affine hyperplane permits encoding the regression intercept
# in a weight vector without changing any bias.
ln_gamma210 = load_original("core.prelude_norm.weight").numpy().astype(np.float64)
ln_beta210 = load_original("core.prelude_norm.bias").numpy().astype(np.float64)
constant_direction210 = 1.0 / ln_gamma210
constant210 = float(ln_beta210 @ constant_direction210)
assert abs(constant210) > 1e-6, constant210
linear210 = coef210[:192] / sd147
intercept210 = float(coef210[192] - mu147 @ linear210)
insulin_W210 = linear210 + (intercept210 / constant210) * constant_direction210
mapping_error210 = np.max(np.abs(
    raw_E210 @ insulin_W210 - design210 @ coef210))
assert mapping_error210 < 2e-4, mapping_error210

base_B210 = load_original("core.b_in.weight").numpy().astype(np.float64)
fallback_B210 = load("core.b_in.weight").numpy().copy()
directions210 = np.linalg.lstsq(
    candidate_W204.astype(np.float64),
    (fallback_B210.astype(np.float64) - base_B210).T, rcond=None
)[0].T
factor_error210 = np.max(np.abs(
    base_B210 + directions210 @ candidate_W204.T - fallback_B210))
assert factor_error210 < 2e-6, factor_error210

candidate_W210 = candidate_W204.astype(np.float64).copy()
candidate_W210[:, 0] = insulin_W210
candidate_B210 = (
    base_B210 + directions210 @ candidate_W210.T
).astype(np.float32)
constraint_error210 = np.max(np.abs(
    Ains206 @ coef210 - targets206[("preserve_half", "preserve_cross")]))

print("standardization / affine mapping / factorization errors",
      standardization_error210, mapping_error210, factor_error210)
print("constraint error", constraint_error210)
print("NE decoder unchanged",
      np.array_equal(candidate_W210[:, 1], candidate_W204[:, 1]))
print("maximum weight change from current",
      np.max(np.abs(candidate_B210 - fallback_B210)))
print("selected proxy diagnostics",
      table206.loc[
          (table206["mode"] == "preserve_half/preserve_cross") &
          (table206["pw"] == 0.003) &
          (table206["ridge"] == 1e-6)
      ].to_dict("records"))

save_edit("core.b_in.weight", torch.from_numpy(candidate_B210))
print("Installed edits", edited())
print("Trial hash",
      hashlib.sha256(candidate_B210.tobytes()).hexdigest()[:16])
print("Difficult-cohort endpoints / count",
      test_stays204[0], test_stays204[-1], len(test_stays204))

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs209–210: sensitivity-preserving insulin trial\n"
            "Run209 inspection failed before any edit. In run210, installed "
            "constrained preserve-half/preserve-cross insulin fit (0.003,1e-6); "
            "NE decoder remains candidate_W204. Verified standardization, "
            "LayerNorm affine conversion, and rank-two factorization before "
            "saving only core.b_in.weight. Exact fallback is fallback_B210. "
            "Proxy RMS old/new about 0.0830/0.0830 versus current 0.1060/0.1022. "
            "Direct factual NLL and dose rollouts are still required.\n")
```