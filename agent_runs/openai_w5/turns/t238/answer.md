The corrected test shows mostly linear, drug-specific signals, but insulin’s signal saturates above roughly 11.5 units/hour. That could weaken high-dose predictions, though it does not explain the unusually weak halving response. I’ll check the centering term next: if it tracks the edited dose rather than pre-treatment history, it can cancel the intended causal correction.

```python
sensor_neurons = {
    "glucose_finger": (107, 7),
    "glucose_lab": (74, 171),
    "map_arterial": (109, 10),
    "map_cuff": (64, 124),
}
print("=== CAUSAL MINUS CONTROL NEURONS ===")
for _event, (_j, _control) in sensor_neurons.items():
    _dw = W["attr_heads.value_bin.0.weight"][_j] - W["attr_heads.value_bin.0.weight"][_control]
    _db = W["attr_heads.value_bin.0.bias"][_j] - W["attr_heads.value_bin.0.bias"][_control]
    print(_event, "ordinary-centering norm",
          round(float(torch.linalg.vector_norm(_dw[ordinary_idx])), 6),
          "private coefficients", _dw[priv].numpy(),
          "bias difference", round(float(_db), 6))

print("\n=== NUMERIC NORMALIZATIONS ===")
for _key, _spec in num_spec.items():
    print(_key, {k: v for k, v in _spec.items()
                 if k in ["median", "iqr", "fine_bin_edges"]})

print("\n=== HEAD/PRIOR SHAPES ===")
print([(n, tuple(W[n].shape)) for n in NAMES
       if n.startswith(("z_prior.", "z_head.", "gap.", "last_numeric.",
                         "last_status.", "type_emb."))])

state_ids = np.unique(state_file["stay_id"])
print("\nState stays:", len(state_ids), state_ids[:8], state_ids[-3:])
print("Train/tuning overlap:",
      len(np.intersect1d(state_ids, train.stay_id.unique())),
      len(np.intersect1d(state_ids, tuning.stay_id.unique())))
print("State hour examples:", np.unique(state_file["hour"])[:20])

print("\n=== RECORDED RATE SUPPORT ===")
for _event in ["insulin", "norepinephrine"]:
    _r = train.loc[train.event.eq(_event), "rate"].dropna()
    print(_event, "n", len(_r), "quantiles",
          _r.quantile([0, .1, .5, .9, .99, 1]).to_dict())
    if _event == "insulin":
        print("fraction whose doubled rate exceeds channel saturation:",
              float((_r * 2 > 11.48).mean()))

print("\n=== ROLLOUT FILE SCHEMA ===")
dose18 = pd.read_csv("/work/fwd_18/dose_edit_edited.csv")
print(dose18.columns.tolist())
print(dose18.head(5).to_string(index=False))

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Runs 209–211: recovered finite-rate channels\n"
             "- Correct clean pairs are insulin (21,61), NE (15,66); initial diagnostic pairing was wrong.\n"
             "- Encoder signals are mostly linear and drug-specific. Insulin saturates near rate 11.49 due to input clipping.\n"
             "- Saturation alone cannot explain doubling response exceeding twice the halving response.\n"
             "- Next hypothesis: ordinary-feature exposure centering may absorb dose interventions; not yet established.\n")
```