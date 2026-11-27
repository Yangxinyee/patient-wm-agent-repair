The cached design uses an intercept in its first column, not its last. That explains the failed check, and the weights remain unchanged. I’ll use the correct layout, verify both the predictions and weight factorization, then install the trial. It succeeds only if direct factual NLL improves without materially weakening the four rollout responses.

```python
key212 = (("preserve_half", "preserve_cross"), (0.003, 1e-6))
coef212 = np.asarray(constrained_fits206[key212], dtype=np.float64)
raw_E212 = np.asarray(Efull134, dtype=np.float64).reshape(-1, 192)
design212 = np.asarray(X147, dtype=np.float64)

standardization_error212 = np.max(np.abs(
    design212[:, 1:] - (raw_E212 - mu147) / sd147))
assert standardization_error212 < 2e-4, standardization_error212
assert np.max(np.abs(design212[:, 0] - 1)) < 1e-6

ln_gamma212 = load_original("core.prelude_norm.weight").numpy().astype(np.float64)
ln_beta212 = load_original("core.prelude_norm.bias").numpy().astype(np.float64)
constant_direction212 = 1.0 / ln_gamma212
constant212 = float(ln_beta212 @ constant_direction212)
assert abs(constant212) > 1e-6
linear212 = coef212[1:] / sd147
intercept212 = float(coef212[0] - mu147 @ linear212)
insulin_W212 = linear212 + (intercept212 / constant212) * constant_direction212
mapping_error212 = np.max(np.abs(
    raw_E212 @ insulin_W212 - design212 @ coef212))
assert mapping_error212 < 2e-4, mapping_error212

base_B212 = load_original("core.b_in.weight").numpy().astype(np.float64)
fallback_B212 = load("core.b_in.weight").numpy().copy()
directions212 = np.linalg.lstsq(
    candidate_W204.astype(np.float64),
    (fallback_B212.astype(np.float64) - base_B212).T, rcond=None
)[0].T
factor_error212 = np.max(np.abs(
    base_B212 + directions212 @ candidate_W204.T - fallback_B212))
assert factor_error212 < 2e-6, factor_error212

candidate_W212 = candidate_W204.astype(np.float64).copy()
candidate_W212[:, 0] = insulin_W212
candidate_B212 = (
    base_B212 + directions212 @ candidate_W212.T
).astype(np.float32)
constraint_error212 = np.max(np.abs(
    Ains206 @ coef212 - targets206[("preserve_half", "preserve_cross")]))
assert constraint_error212 < 1e-7, constraint_error212

print("standardization / affine mapping / factorization errors",
      standardization_error212, mapping_error212, factor_error212)
print("constraint error", constraint_error212)
print("NE decoder unchanged",
      np.array_equal(candidate_W212[:, 1], candidate_W204[:, 1]))
print("maximum weight change", np.max(np.abs(candidate_B212 - fallback_B212)))
print("selected proxy diagnostics",
      table206.loc[
          (table206["mode"] == "preserve_half/preserve_cross") &
          (table206["pw"] == 0.003) &
          (table206["ridge"] == 1e-6)
      ].to_dict("records"))

save_edit("core.b_in.weight", torch.from_numpy(candidate_B212))
print("Installed edits", edited())
print("Trial hash",
      hashlib.sha256(candidate_B212.tobytes()).hexdigest()[:16])
print("Difficult-cohort endpoints / count",
      test_stays204[0], test_stays204[-1], len(test_stays204))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run212: constrained insulin trial installed\n"
            "Recovered intercept-FIRST design convention. Verified exact "
            "cached standardization, LayerNorm affine folding and rank-two "
            "factorization. Installed preserve-half/preserve-cross insulin "
            "fit (0.003,1e-6), retaining candidate_W204 NE decoder. Only "
            "core.b_in.weight edited. Exact fallback: fallback_B212. "
            "Candidate requires paired factual NLL and rollout validation.\n")
```