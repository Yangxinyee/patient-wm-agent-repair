The rollout files include the recorded-dose baseline as well as the treatment shift, so we can check whether the stronger graft mainly distorted both paths. I’ll compare hourly results and recover the identification diagnostics. Large baseline changes with little extra treatment separation would favor redesigning the decoder rather than increasing its strength.

```python
print("=== INSULIN HOURLY DIAGNOSIS ===")
_roll2 = pd.read_csv("/work/fwd_9/dose_edit_edited.csv")
_roll24 = pd.read_csv("/work/fwd_11/dose_edit_edited.csv")
_hourly = []
for _label, _df in [("beta2", _roll2), ("beta24", _roll24)]:
    _s = _df[(_df["edit"] == "insulin_x2") & _df["reached"]]
    _q = _s.groupby("hour")[["shift", "factual"]].mean()
    _q.columns = [f"{c}_{_label}" for c in _q.columns]
    _hourly.append(_q)
print(pd.concat(_hourly, axis=1).round(3).to_string())

print("=== RECOVERED THRESHOLD IDENTIFICATION ===")
print(section[:5300] if isinstance(section, str) else repr(section)[:5300])
print("=== NE DML RESULTS ===")
print(repr(ne_dml_results)[:1400])
print("=== RD DYNAMIC PARAMETERS ===")
print(rd_dynamic_fit["x"])
print("=== IMPLIED INSULIN TARGETS ===")
_rd_beta, _rd_rho = map(float, rd_dynamic_fit["x"])
print("train:", causal_shift_table(P, train_ids, "glucose_finger", "insulin", _rd_beta, _rd_rho))
print("first 100 tuning:", causal_shift_table(TP[:100], tuning_ids[:100],
      "glucose_finger", "insulin", _rd_beta, _rd_rho))
print("=== GRAFT REPRESENTATION ===")
print("nodes:", graft_nodes, "gate scale:", graft_gate_scale)
for _n in graft_names:
    print(_n, type(graft_zero[_n]).__name__,
          tuple(graft_zero[_n].shape), "unit norm", float(np.linalg.norm(graft_unit[_n])))
```
