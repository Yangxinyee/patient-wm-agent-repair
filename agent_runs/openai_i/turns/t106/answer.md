The run stopped because the saved action cases contain more fields than expected; it did not validate the features or change any weights. The dose residuals do align with the full training panel. I’ll fix the case parsing and rerun the isolated-start check before fitting anything.

```python
_case_by_idx98 = {int(c[0]): c[1] for c in action_cases50}
_clean_rows98 = [_case_by_idx98[int(i)] for i in clean_indices50]
_clean_drugs98 = np.array([rows[0]["event"] for rows in _clean_rows98])
_clean_target98 = torch.from_numpy(latent_state[clean_indices50, -32:]).float()
_unit_table98 = repair_weights["embed.categorical.rate_unit.weight"]

print("First action-case fields:",
      [(type(x).__name__, str(x)[:90]) for x in action_cases50[0]])

_results98 = []
for _ri98 in range(4):
    for _rn98 in range(4):
        _us98 = []
        for _rows98 in _clean_rows98:
            _row98 = _rows98[0]
            _new_unit98 = _ri98 if _row98["event"] == "insulin" else _rn98
            _atom98 = (latent_start_atom(_row98)
                       - _unit_table98[2] + _unit_table98[_new_unit98])
            _us98.append(latent_action_encode([_atom98]))
        _us98 = torch.stack(_us98)
        _hh98 = latent_hidden(clean_ctx50, _us98)
        _zz98 = repair_linear(_hh98, "z_post.2")[:, :32]
        _rms98 = torch.sqrt(torch.mean((_zz98 - _clean_target98) ** 2, dim=1)).numpy()
        _results98.append({
            "insulin_rate_unit": _ri98, "ne_rate_unit": _rn98,
            "rms": float(_rms98.mean()),
            "insulin_rms": float(_rms98[_clean_drugs98 == "insulin"].mean()),
            "ne_rms": float(_rms98[_clean_drugs98 == "norepinephrine"].mean())
        })
_validation98 = pd.DataFrame(_results98).sort_values("rms")
print("\nPOSTERIOR VALIDATION")
print(_validation98.head(6).to_string(index=False))

print("\nFULL FEATURE RECONSTRUCTION")
print("hidden_all65", tuple(hidden_all65.shape), "target", tuple(target_z65.shape))
print("RMS:", float(torch.sqrt(torch.mean(
    (repair_linear(hidden_all65, "z_post.2")[:, :32] - target_z65) ** 2))))

print("\nACTION INNOVATION ALIGNMENT")
_action_panel_pos98 = panel.index.get_indexer(pd.MultiIndex.from_arrays([
    latent_sid[action_indices50], np.floor(latent_hour[action_indices50]).astype(int)
], names=panel.index.names))
assert np.all(_action_panel_pos98 >= 0)
for _j98, _drug98 in enumerate(("insulin", "norepinephrine")):
    _mask98 = rates50[:, _j98] > 0
    _innov98 = dose_residuals[_drug98][_action_panel_pos98]
    _valid98 = _mask98 & np.isfinite(_innov98)
    print(_drug98, "active", int(_mask98.sum()), "residual matched", int(_valid98.sum()),
          "rate mean", float(rates50[_valid98, _j98].mean()),
          "innovation mean/sd",
          float(_innov98[_valid98].mean()), float(_innov98[_valid98].std()))

print("\nCAUSAL POLICY FEATURE COLUMNS")
print(list(HC.columns))
with open("/work/notes.md", "a") as _f:
    _f.write("\n## Python 98 — corrected isolated-action validation\n"
             "- Python 97 stopped on action-case tuple arity, before any validation or edit.\n"
             "- Repeated the validation with indexed case fields and aligned active-dose "
             "innovations to the training panel.\n")
```