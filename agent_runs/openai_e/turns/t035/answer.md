Standalone drug starts match to numerical precision, but bundles containing infusion endings do not. There are too few verified rows for a reliable probe. I’ll inspect the ending convention and test the larger, approximate reconstruction as a diagnostic—not as proof. The approach remains viable only if held-out innovations are recoverable with little non-drug leakage.

```python
print("=== DRUG BUNDLE TYPES ===")
for _label32, _ix32 in sorted(_groups31.items(), key=lambda x: -len(x[1])):
    if any(d in str(_label32) for d in ("insulin", "norepinephrine")):
        print(_label32, len(_ix32),
              "posterior RMSE", round(float(np.sqrt(np.mean(_err31[_ix32]**2))), 5))

print("=== EMBEDDING TABLE SHAPES ===")
print({n: SHAPES[n] for n in NAMES
       if n.startswith("embed.") and any(s in n for s in
          ("categorical", "kind", "ekind", "kq", "text"))})

# Diagnostic probes: factual dose innovations at action rows, zero elsewhere.
_T32 = np.zeros((len(_H31), 2), dtype=np.float64)
_T32[action_rows] = action_targets
_isaction32 = np.zeros(len(_H31), dtype=bool)
_isaction32[action_rows] = True
_sid32 = S0["stay_id"]
_tr32 = _sid32 % 5 != 0
_te32 = ~_tr32
_mu32 = _H31[_tr32].mean(axis=0)
_sd32 = np.maximum(_H31[_tr32].std(axis=0), 0.05)
_X32 = (_H31 - _mu32) / _sd32
_probe32_candidates = []
_results32 = []
for _other_weight32 in (1.0, 10.0):
    _weights32 = np.where(_isaction32, 1.0, _other_weight32)
    for _alpha32 in (1.0, 10.0, 100.0):
        _model32 = Ridge(alpha=_alpha32).fit(
            _X32[_tr32], _T32[_tr32],
            sample_weight=_weights32[_tr32])
        _prediction32 = _model32.predict(_X32)
        _a32 = _te32 & _isaction32
        _n32 = _te32 & ~_isaction32
        _v32 = _a32 & (_err31 < 0.002)
        _row32 = {
            "other_weight": _other_weight32, "alpha": _alpha32,
            "action_R2": np.round(r2_score(
                _T32[_a32], _prediction32[_a32],
                multioutput="raw_values"), 4).tolist(),
            "non_action_RMS": np.round(np.sqrt(np.mean(
                _prediction32[_n32]**2, axis=0)), 4).tolist(),
            "verified_action_RMSE": np.round(np.sqrt(np.mean(
                (_prediction32[_v32]-_T32[_v32])**2, axis=0)), 4).tolist()
        }
        _probe32_candidates.append((_model32, _mu32.copy(), _sd32.copy(), _row32))
        _results32.append(_row32)
print("=== APPROXIMATE POSTERIOR PROBES ===")
print(_results32)

print("=== EXAMPLE SINGLE-END RECORD ===")
for _label32, _ix32 in _groups31.items():
    if _label32[0] == "none" and _label32[1] in ("insulin", "norepinephrine"):
        _j32 = _ix32[0]
        _key32 = (int(S0["stay_id"][_j32]), int(round(S0["hour"][_j32]*60)))
        print("row", _j32, "type", _label32,
              "records", record_ends[_key32][:1])
        break

with open("/work/notes.md", "a") as _f32:
    _f32.write(
        "\n## Run 32: ending audit and approximate innovation probes\n"
        "Standalone drug starts are numerically verified, but repeated-hour "
        "bundles with END atoms are not. Tested larger approximate hidden-feature "
        "probes with explicit non-action zero constraints; diagnostic only.\n"
        f"Results: {_results32}.\n"
        "No edits. Python runs: 32; forward passes: 3.\n"
    )
```