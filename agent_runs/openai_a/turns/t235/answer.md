The edit uses a glucose-finger gate and a dose-sensitive contrast learned from model states. I’ll screen stronger versions of the existing statistical fits without changing the current weights. A candidate with larger dose contrasts at comparable cached factual cost would justify another GPU test.

```python
_screen_fact = {k: selective_cache(v, 30.0) for k, v in sc_fact.items()}
_screen_syn = {k: selective_cache(v, 30.0) for k, v in sc_syn.items()}

print("Available fit keys:", list(r3_candidates))
print("Cache shapes:", {
    k: (v["x"].shape, v["gate"].shape, float(v["original_nll"]))
    for k, v in _screen_fact.items()
})

stronger_screen = []
for _lam, _scales in [
    (0.3, [6.0, 7.0, 8.0, 9.0, 10.0]),
    (1.0, [10.0, 14.0, 18.0, 22.0, 26.0]),
]:
    if (_lam, 1.0) not in r3_candidates:
        continue
    _fit = r3_candidates[(_lam, 1.0)]
    for _scale in _scales:
        _row = {"lam": _lam, "scale": _scale}
        for _split, _cache in _screen_fact.items():
            _row[_split] = float(
                cache_losses(_cache, _fit["w"], _fit["b"], _scale)
                - _cache["original_nll"]
            )
        _means = {
            k: sc_evaluate(v, _fit["w"], _fit["b"], _scale)["mean"]
            for k, v in _screen_syn.items()
        }
        _row["half"] = float(_means[0.5] - _means[1.0])
        _row["double"] = float(_means[2.0] - _means[1.0])
        stronger_screen.append(_row)

stronger_screen_df = pd.DataFrame(stronger_screen)
print(stronger_screen_df.round(6).to_string(index=False))
print("Weights unchanged:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Stronger-fit CPU screen; Python 205, forwards 21/50\n")
    f.write("- Screen only; stronger6_checkpoint remains installed.\n")
    f.write(stronger_screen_df.to_string(index=False) + "\n")
```
