The screen stopped because the loss helper returns per-reading values, not a scalar. No weights changed. I’ll average those losses and rerun the comparison; the existing scale-6 result should reproduce its earlier cached cost before I trust the new rows.

```python
stronger_screen = []
for _lam, _scales in [
    (0.3, [6.0, 7.0, 8.0, 9.0, 10.0]),
    (1.0, [10.0, 14.0, 18.0, 22.0, 26.0]),
]:
    _fit = r3_candidates[(_lam, 1.0)]
    for _scale in _scales:
        _row = {"lam": _lam, "scale": _scale}
        for _split, _cache in _screen_fact.items():
            _loss = np.asarray(
                cache_losses(_cache, _fit["w"], _fit["b"], _scale)
            )
            _row[_split] = float(_loss.mean() - _cache["original_nll"])
        _means = {
            k: float(sc_evaluate(v, _fit["w"], _fit["b"], _scale)["mean"])
            for k, v in _screen_syn.items()
        }
        _row["half"] = _means[0.5] - _means[1.0]
        _row["double"] = _means[2.0] - _means[1.0]
        stronger_screen.append(_row)

stronger_screen_df = pd.DataFrame(stronger_screen)
print(stronger_screen_df.round(6).to_string(index=False))
print("Weights unchanged:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Corrected stronger-fit CPU screen; Python 206, forwards 21/50\n")
    f.write("- Run 205 stopped on array-to-scalar conversion; no edits changed.\n")
    f.write(stronger_screen_df.to_string(index=False) + "\n")
```
