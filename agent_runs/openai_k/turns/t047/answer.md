The nonlinear center preserves dose-response slopes, but its norepinephrine error is still too large for a safe factual correction. Before editing, I’ll verify the innovation-to-bundle alignment and inspect the prediction registers. Correct alignment and a locally validated readout are needed to distinguish a real treatment correction from a modeling artifact.

```python
print("=== VOCABULARY STRUCTURE ===")
print("top-level keys:", list(VOCAB))
for rr42_key in ["config", "codes", "event_kinds", "kinds", "categorical",
                 "categorical_values", "value_bin_centres"]:
    if rr42_key in VOCAB:
        print(rr42_key, str(VOCAB[rr42_key])[:2200])
print("\n=== TARGET ALIGNMENT ===")
print("P index:", type(P.index), "columns:", list(P.columns))
print(P.head(3).to_string())
print("Xc:", type(Xc), getattr(Xc, "shape", None))
print("T_all:", type(T_all), getattr(T_all, "shape", None))
print("innov_targets difference RMS:",
      np.sqrt(np.mean((np.asarray(innov_targets_audit) - rr41_innov)**2, axis=0)))
for rr42_j, rr42_drug in enumerate(["insulin", "norepinephrine"]):
    for rr42_i in np.flatnonzero(rr39_rates[:, rr42_j] > 0)[:3]:
        rr42_key = state_keys_audit[rr42_i]
        print(rr42_key, "rates:", rr39_rates[rr42_i],
              "position:", rr41_cp[rr42_i],
              "innovation:", np.asarray(innov_targets_audit)[rr42_i])
        print([(r.event, float(r.rate) if pd.notna(r.rate) else None, int(kind))
               for r, kind in bundle_records_audit[rr42_key]])
print("\n=== FACTUAL AND STATE LAYOUT ===")
print("train factual:", train_factual0.shape, train_factual0.head(3).to_string(index=False))
print("state x shape:", train_states0["x"].shape)
print("probe indices:", core_probe_indices[:12])
print("WORLD conventions:",
      {n: globals().get(n) for n in ["world_kq_audit", "world_bin_side_audit",
                                     "end_kq_audit", "glu_unit_audit",
                                     "map_unit_audit"]})
print("Helper primitive names:",
      {n: globals()[n].__code__.co_names for n in
       ["lin0", "norm0", "mlp0", "repaired_core_local"]})
print("\n=== EARLIER RECONSTRUCTION NOTES ===")
rr42_notes = Path("/work/notes.md").read_text()
print(rr42_notes[rr42_notes.find("## Run 17:"):rr42_notes.find("## Runs 23")][:2600])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 42: target-alignment and prediction-readout audit\n"
            "- Dose-invariant centering retains intervention slopes, but norepinephrine centering error remains large.\n"
            "- Checking innovation alignment and head/register layout before committing an edit.\n"
            "- Python runs: 42; forward passes: 4; no tensors edited.\n")
```