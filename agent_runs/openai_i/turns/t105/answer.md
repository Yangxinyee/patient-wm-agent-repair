The preserved action features align with 100 training stays, and the two drugs have very different posterior sensitivities. I’ll validate those features on isolated infusion starts and check the saved dose residuals. Close agreement with the exposed posterior means would justify fitting a centered edit; poor agreement would rule out using these features as-is.

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

print("DOSE RESIDUAL ALIGNMENT")
print("panel index:", panel.index.names, list(panel.index[:3]))
for _drug97, _obj97 in dose_residuals.items():
    print(_drug97, type(_obj97).__name__,
          getattr(_obj97, "shape", None),
          "finite", int(np.isfinite(np.asarray(_obj97)).sum()))
    if isinstance(_obj97, pd.Series):
        print("index:", _obj97.index.names, list(_obj97.index[:3]))

print("\nISOLATED-START POSTERIOR VALIDATION")
_case_by_idx97 = {int(i): rows for i, rows in action_cases50}
_clean_rows97 = [_case_by_idx97[int(i)] for i in clean_indices50]
_clean_drugs97 = np.array([rows[0]["event"] for rows in _clean_rows97])
_clean_target97 = torch.from_numpy(latent_state[clean_indices50, -32:]).float()
_unit_table97 = repair_weights["embed.categorical.rate_unit.weight"]

_results97 = []
for _ri97 in range(4):
    for _rn97 in range(4):
        _atoms97 = []
        for _rows97 in _clean_rows97:
            _row97 = _rows97[0]
            _old_atom97 = latent_start_atom(_row97)
            _new_unit97 = _ri97 if _row97["event"] == "insulin" else _rn97
            _atoms97.append(_old_atom97 - _unit_table97[2] + _unit_table97[_new_unit97])
        _us97 = torch.stack([latent_action_encode([a]) for a in _atoms97])
        _hh97 = latent_hidden(clean_ctx50, _us97)
        _zz97 = repair_linear(_hh97, "z_post.2")[:, :32]
        _rms97 = torch.sqrt(torch.mean((_zz97 - _clean_target97) ** 2, dim=1)).numpy()
        _results97.append({
            "insulin_rate_unit": _ri97,
            "ne_rate_unit": _rn97,
            "rms": float(np.mean(_rms97)),
            "insulin_rms": float(np.mean(_rms97[_clean_drugs97 == "insulin"])),
            "ne_rms": float(np.mean(_rms97[_clean_drugs97 == "norepinephrine"]))
        })
_validation97 = pd.DataFrame(_results97).sort_values("rms")
print(_validation97.head(6).to_string(index=False))

print("\nFACTUAL POSTERIOR FEATURE ARRAY")
print("hidden_all65", tuple(hidden_all65.shape),
      "target_z65", tuple(target_z65.shape),
      "contexts65", tuple(contexts65.shape))
print("reconstructed mean RMS:",
      float(torch.sqrt(torch.mean(
          (repair_linear(hidden_all65, "z_post.2")[:, :32] - target_z65) ** 2))))

print("\nCAUSAL POLICY FEATURES")
print("HC columns:", list(HC.columns))
print("clean start counts:", pd.Series(_clean_drugs97).value_counts().to_dict())

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Python 97 — isolated-action validation\n"
             "- Checking a finite set of rate-unit conventions against exposed original "
             "posterior means, and residual indexing against the hourly training panel.\n"
             "- These checks gate the posterior feature fit; no weights changed.\n")
```