The empty-action hypothesis is refuted: those vectors are already zero. Drug-bundle reconstruction remains accurate, so I’ll now isolate observation embeddings and their time alignment. A consistent error pattern by event type or interval would identify which convention needs correction.

```python
print("=== VOCABULARY AND EMBEDDING CONVENTIONS ===")
print("VOCAB keys:", list(VOCAB))
for key92 in VOCAB:
    if any(s in key92.lower() for s in ("categor", "unit", "kind", "normal")):
        text92 = repr(VOCAB[key92])
        print(key92, text92[:2200])
print("unit_ids_repair:", unit_ids_repair)
print("rate_unit_ids_repair:", rate_unit_ids_repair)
print("measurement_events:", measurement_events)
for name92 in NAMES:
    if name92.startswith(("embed.categorical", "embed.kind", "embed.ekind")):
        print(name92, SHAPES[name92])
print("Atom helper constants:", embed_record_atom.__code__.co_consts)
print("Atom helper variables:", embed_record_atom.__code__.co_varnames)

print("\n=== ERROR BY TIME INTERVAL AND BUNDLE SIZE ===")
meta92 = sm_repair.iloc[rows91].reset_index(drop=True).copy()
meta92["dt_seconds"] = (
    sm_repair.iloc[rows91]["hour"].to_numpy()
    - sm_repair.iloc[rows91 - 1]["hour"].to_numpy()
) * 3600
meta92["error"] = posterior_error_repair
meta92["has_action"] = mask91
print(meta92.groupby(["has_action", "dt_seconds", "n_atoms"])["error"]
      .agg(["count", "mean", "max"]).reset_index().round(6).to_string(index=False))

print("\n=== RAW RECORD EXAMPLES ===")
print(reference_records_repair.head(12).to_string(index=False))
print("\n=== EXISTING OBSERVATION AUDIT ===")
for item92 in obs_checks_repair[:4]:
    print(repr(item92)[:1000])
print("\n=== SET STRUCTURES ===")
for i92 in np.flatnonzero(~mask91)[:3]:
    print("index", int(i92),
          "obs set type", type(obs_sets_repair[i92]).__name__,
          "obs set length", len(obs_sets_repair[i92]),
          "action records", repr(action_records_repair[i92])[:300])
print("Current edits:", edited())
```
