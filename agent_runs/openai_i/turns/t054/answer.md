The kernel still has the hourly panel, dose-residual models, latent states, and head-reconstruction helpers. The display was truncated by a function’s tensor-valued defaults, so I’ll print only targeted summaries. Next I need the estimated response over time and the posterior layer shapes; these will tell us whether a low-rank recurrent correction is feasible.

```python
print("=== REMAINING VARIABLE NAMES ===")
print(", ".join(sorted(
    n for n in globals()
    if not n.startswith("_") and n > "linear"
)))

print("\n=== PANEL COLUMNS ===")
print("H:", list(H.columns))
print("HC:", list(HC.columns))
print("head_rows:", list(head_rows.columns))
print("latent_records:", list(latent_records.columns))

print("\n=== EFFECT ESTIMATES ===")
print(dml_table.to_string(index=False))
print(irf_table.to_string(index=False))
print("Impulse coefficients:")
for _key, _value in irf_coefficients.items():
    print(_key, np.asarray(_value).round(5).tolist())

print("\n=== RETAINED MODEL METADATA ===")
for _name in ["dose_models", "dose_residuals", "contexts", "head2_settings",
              "best_head_settings", "future_actions"]:
    _obj = globals().get(_name)
    if isinstance(_obj, dict):
        print(_name, {str(k): type(v).__name__ for k, v in _obj.items()})
for _name in ["latent_context", "latent_hidden", "latent_post_mean",
              "latent_start_atom", "latent_encode_batch", "latent_action_encode",
              "head_exact"]:
    _fn = globals().get(_name)
    if callable(_fn):
        print(_name, "arguments:", list(inspect.signature(_fn).parameters))

print("\n=== RELEVANT TENSOR SHAPES ===")
for _name in NAMES:
    if any(_s in _name for _s in
           ("z_post", "z_prior", "z_in", "z_head", "encode_action", "attr_heads.value_bin")):
        print(_name, SHAPES[_name])

print("\n=== IDENTIFICATION NOTEBOOK EXCERPTS ===")
_notes = Path("/work/notes.md").read_text()
_sections = re.split(r"(?=^## )", _notes, flags=re.M)
_limit = 2000
for _section in _sections:
    if any(_word in _section.lower() for _word in ("discontinuity", "placebo", "impulse")):
        _piece = _section[:min(900, _limit)]
        print(_piece)
        _limit -= len(_piece)
        if _limit <= 0:
            break
```
