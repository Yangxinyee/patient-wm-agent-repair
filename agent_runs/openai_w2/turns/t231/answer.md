The corrected posterior inputs are ready. The prelude helper returns both raw and normalized slot features, so I’ll identify the exact features read by `b_in` and recover the existing regression’s scaling. This will confirm that the new cohort can be added to the factual constraints without inadvertently changing the edit’s output directions.

```python
print("PRELUDE RETURN INSTRUCTIONS")
for r199_i in list(dis.get_instructions(prelude_features111))[-24:]:
    print(r199_i.opname, r199_i.argrepr)

print("\nDECODER FIT INSTRUCTIONS")
for r199_i in dis.get_instructions(fit_zero_fact171):
    print(r199_i.opname, r199_i.argrepr)

print("\nFOLDING INSTRUCTIONS")
for r199_i in dis.get_instructions(fold_decoder147):
    print(r199_i.opname, r199_i.argrepr)

print("\nPRELUDE FEATURE SUMMARIES")
for r199_k, r199_v in enumerate(r198_prelude):
    print(r199_k, tuple(r199_v.shape),
          "mean", float(r199_v.mean()), "sd", float(r199_v.std()))
print("saved factual E:", F147.shape,
      "mean", float(F147.mean()), "sd", float(F147.std()))

print("\nSTANDARDIZATION CHECK")
r199_old_flat = np.asarray(F147, dtype=np.float64).reshape(-1, 192)
r199_X_rebuilt = np.column_stack([
    (r199_old_flat - mu147) / sd147,
    np.ones(len(r199_old_flat))
])
print("rebuilt vs X147:", float(np.max(np.abs(r199_X_rebuilt - X147))))
print("Xcheck171 vs X147:", float(np.max(np.abs(Xcheck171 - X147))))
r199_H_raw = Xcheck171.T @ Xcheck171
r199_H_scale = float(np.trace(Hf171) / np.trace(r199_H_raw))
print("Hf171 normalization factor:", r199_H_scale)
print("Hf171 gram agreement:",
      float(np.max(np.abs(Hf171 - r199_H_scale * r199_H_raw))))
print("probe matrix/target summaries:",
      D147.shape, Y147.shape,
      np.mean(Y147, axis=0), np.std(Y147, axis=0))
print("probe normalizations:",
      float(np.trace(Hp171) / np.trace(D147.T @ D147)))

print("\nRECOVER EXACT EDIT DIRECTIONS")
r199_B0 = np.asarray(load_original("core.b_in.weight"), dtype=np.float64)
r199_W = np.asarray(validated_W176, dtype=np.float64)
r199_delta = np.asarray(validated_B176, dtype=np.float64) - r199_B0
r199_L = r199_delta @ r199_W @ np.linalg.inv(r199_W.T @ r199_W)
print("rank-two reconstruction maximum error:",
      float(np.max(np.abs(r199_delta - r199_L @ r199_W.T))))
print("current edit unchanged:",
      np.array_equal(load("core.b_in.weight"), validated_B176))
```
