The state request returned 4,035 bundles, but no new evidence about drug effects. I’ll check whether those states retain the history-adjusted dose innovations estimated from the records, and inspect the prediction-side tensors. Strong held-out recovery would support a readout-based repair; weak recovery would suggest that the missing dose response must be repaired earlier in the model.

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

train_states0 = dict(np.load("/work/fwd_3/states_original.npz"))
print("=== STATE ARRAYS ===")
for k, v in train_states0.items():
    print(k, v.shape, v.dtype)
print(pd.DataFrame({k: v[:12] for k, v in train_states0.items() if k != "x"}).to_string(index=False))

print("\n=== PREDICTION-SIDE TENSORS ===")
for nm in NAMES:
    if (nm.startswith("core.") and not any(t in nm for t in
            (".prelude.", ".prelude_norm.", ".blocks.", ".coda."))) or nm.startswith(
            ("last_", "embed.categorical.", "embed.kind.", "embed.ekind.",
             "attr_heads.value_bin.", "query_norm.", "type_emb.")):
        print(nm, SHAPES[nm])

print("\n=== NUMERIC TABLES ===")
for item in VOCAB["numeric"]:
    print(item["code"], item["attribute"], "median", item["median"],
          "iqr", item["iqr"], "edges", item["fine_bin_edges"])
print("Value centres:", VOCAB["value_bin_centres"])
print("Load return type:", type(load_original("attr_heads.value_bin.2.weight")).__name__)
print("Causal-row mapping:", causal_rows.dtype, causal_rows[:6], hours[causal_rows[:6]])

state_drug_rows0 = np.flatnonzero(
    (train_states0["has_insulin"].astype(bool) |
     train_states0["has_norepinephrine"].astype(bool)) &
    (train_states0["hour"] >= 0) & (train_states0["hour"] < 24)
)
state_drug_panel0 = P.index.get_indexer(pd.MultiIndex.from_arrays([
    train_states0["stay_id"][state_drug_rows0],
    np.floor(train_states0["hour"][state_drug_rows0]).astype(int)
]))
assert np.all(state_drug_panel0 >= 0)
causal_position = np.full(len(P), -1, dtype=np.int32)
causal_position[causal_rows] = np.arange(len(causal_rows))
state_drug_cpos0 = causal_position[state_drug_panel0]
assert np.all(state_drug_cpos0 >= 0)
state_drug_innov0 = dose_resid[state_drug_cpos0]
state_drug_rate0 = T_all[state_drug_panel0]
state_fold0 = (train_states0["stay_id"][state_drug_rows0] % 2).astype(int)

state_recovery0 = {}
print("\n=== STAY-HELD-OUT LINEAR DOSE RECOVERY ===")
for label, features, alpha in [
    ("out", train_states0["x"][state_drug_rows0, :192], 10.0),
    ("out+h+z", train_states0["x"][state_drug_rows0], 100.0)
]:
    for target_name, target in [("actual rate", state_drug_rate0),
                                ("dose innovation", state_drug_innov0)]:
        estimate = np.zeros_like(target, dtype=float)
        for f in (0, 1):
            keep = state_fold0 != f
            reg = Ridge(alpha=alpha).fit(features[keep], target[keep])
            estimate[~keep] = reg.predict(features[~keep])
        scores = r2_score(target, estimate, multioutput="raw_values")
        state_recovery0[(label, target_name)] = scores.tolist()
        print(label, target_name, "n", len(target),
              "R2 insulin / norepinephrine:", np.round(scores, 4))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 9 / forward 3: original training states\n"
        "- Obtained 4,035 post-bundle states from 100 training stays.\n"
        "- Checked whether linear state readouts retain conditionally centered dose innovations.\n"
        "- Recovery diagnostics: " + repr(state_recovery0) + "\n"
        "- No weights edited.\n"
        "- Python runs: 9; forward passes: 3.\n"
    )
```