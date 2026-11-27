The cached factors reconstruct the validated edit within float32 rounding. I’ll now test a 50% half-dose blend, leaving both drug gains unchanged. It would be useful if the half-dose rollout improves without worsening the doubled-dose response or breaching the factual-fit limit.

```python
import hashlib

fallback_B232 = validated_B225.copy()
half_fraction232 = 0.50
coef232 = (
    (1.0 - half_fraction232)*coef_base226
    + half_fraction232*coef_linear226
)
insulin_W232 = decoder_to_weight226(coef232)
mapping_error232 = float(np.max(np.abs(
    E226 @ insulin_W232 - X226 @ coef232
)))
assert mapping_error232 < 2e-5

delta_ins232 = np.outer(V218[:, 0], insulin_W232)
candidate_B232 = (
    original231 + 1.20*delta_ins232 + 1.05*delta_ne218
).astype(np.float32)

assert candidate_B232.shape == tuple(SHAPES["core.b_in.weight"])
save_edit("core.b_in.weight", torch.from_numpy(candidate_B232))
assert np.array_equal(
    numpy231(load("core.b_in.weight")).astype(np.float32), candidate_B232
)
assert edited() == ["core.b_in.weight"]

print("INSTALLED PROVISIONAL TRIAL",
      {"half_fraction": half_fraction232,
       "insulin_gain": 1.20, "NE_gain": 1.05,
       "mapping_error": mapping_error232,
       "hash": hashlib.sha256(candidate_B232.tobytes()).hexdigest()[:12]})
rollout_ids232 = sorted(int(s) for s in roll225["stay_id"].unique())
print("MATCHED ROLLOUT IDS", len(rollout_ids232),
      rollout_ids232[:3], rollout_ids232[-3:])

with open("/work/notes.md", "a") as f232:
    f232.write(
        "\n## Run232: isolated half-dose blend trial installed\n"
        "Only core.b_in.weight edited. Candidate_B232 uses 50% linear-half "
        "decoder blend, retaining insulin gain 1.20 and NE gain 1.05. "
        f"Mapping error {mapping_error232:.3g}. "
        "Exact accepted fallback is validated_B225 / fallback_B232. "
        "Trial is provisional: require paired factual NLL and exact dose rollouts.\n"
    )
```
