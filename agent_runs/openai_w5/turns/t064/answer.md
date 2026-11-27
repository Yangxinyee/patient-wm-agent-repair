The printed rollout targets were too long and hid the useful diagnostics; they remain estimates, not observed truth. I’ll print compact summaries and inspect the correction’s time alignment. Correct indexing and accurate state reconstruction would support testing the existing features; failures there would invalidate the earlier decoding conclusions.

```python
print("Compact estimated targets:",
      {str(k): tuple(v[:2]) for k, v in target_checks_A.items()})
for key in ["zerr4", "row_error"]:
    if key in globals():
        a_repair = np.asarray(globals()[key])
        print(key, np.quantile(a_repair[np.isfinite(a_repair)], [0, .5, .9, 1]).tolist())
for flag in [False, True]:
    a_repair = np.asarray(zerr4)[np.asarray(startmask4) == flag]
    print("latent error action-start", flag, "n", len(a_repair),
          "median/max", float(np.median(a_repair)), float(np.max(a_repair)))
print("Correction index sample:", list(corrTrainA["idx"][:5]))
print("Correction innovation sample:", corrTrainA["innovations"][:5].tolist())
print("Correction desired sample:", corrTrainA["desired"][:5].tolist())
print("Action function signatures:",
      {k: str(inspect.signature(globals()[k]))
       for k in ["action_vectors4", "action_atom_exact", "atom_action_local",
                 "set_encode_single_local", "head_features_layout"]})
print("Existing decoder scalars:",
      {k: float(v) for k, v in list(globals().items())
       if any(s in k.lower() for s in ["r2", "score", "alpha"])
       and isinstance(v, (int, float, np.number))})
print("=== correction_data ===")
print(compact_dis_repair(correction_data)[:6500])
```
