The posterior hidden features are available: 1,312 drug bundles with 192 features each. I’ll inspect the unmatched bundles and the saved perturbation and policy objects. If the failures are explained by boundary or END-only bundles, we can handle them explicitly; if ordinary infusion starts are mismatched, the alignment must be corrected first.

```python
print("=== EARLIER COMPACT INVENTORY ===")
print(_text[:5100])
print("=== ALIGNMENT FAILURES ===")
_bad = np.flatnonzero(panel_pos111 < 0)
print("panel index", panel.index.names, "sample", list(panel.index[:3]))
print("unmatched", len(_bad))
if len(_bad):
    _bi = drug_idx110[_bad]
    print(pd.DataFrame({
        "bundle_index": _bi,
        "stay_id": state_ids[_bi],
        "hour": state_hours[_bi],
        "insulin_rate": drug_rates110[_bad, 0],
        "ne_rate": drug_rates110[_bad, 1],
    }).head(20).to_string(index=False))
print("state IDs", state_ids.min(), state_ids.max(),
      "train IDs", train_ids.min(), train_ids.max())
print("=== SAVED OBJECTS ===")
for _name, _obj in list(globals().items()):
    if _name.startswith("_") or not re.search(
        "case|drug|perturb|hidden|policy|dml|nuisance|innovation|direction|irf", _name, re.I
    ):
        continue
    if isinstance(_obj, dict):
        print(_name, "dict", len(_obj), str(list(_obj.keys())[:12])[:220])
    elif isinstance(_obj, (list, tuple)):
        print(_name, type(_obj).__name__, len(_obj),
              "first-type", type(_obj[0]).__name__ if len(_obj) else "")
    elif isinstance(_obj, (str, int, float)):
        print(_name, repr(_obj)[:120])
print("=== NOTEBOOK IDENTIFICATION HEADINGS ===")
for _line in _notes.splitlines():
    if _line.startswith("##") and re.search(
        "effect|causal|ident|impulse|lag|target|dose|policy|threshold|regression|dynamic", _line, re.I
    ):
        print(_line)
```
