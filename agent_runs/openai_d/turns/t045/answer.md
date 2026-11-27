The held-out decoding test failed for both drugs. That refutes the current head-only correction: amplifying those fitted directions would mostly amplify noise. I’ll look earlier in the action pathway, where the dose enters the model, and test whether the posterior latent or recurrent slots retain information that the reading head loses.

```python
print("=== ACTION / POSTERIOR WEIGHTS ===")
for _n in NAMES:
    if (_n.startswith("encode_action.") or _n.startswith("embed.") or
        _n.startswith("z_post.") or _n.startswith("z_in.") or
        _n.startswith("core.prelude.")):
        print(_n, SHAPES[_n])

print("=== VOCABULARY STRUCTURE ===")
print("keys:", list(VOCAB))
print("config:", VOCAB["config"])
for _k, _v in VOCAB.items():
    if any(_s in _k.lower() for _s in ("categor", "kind", "unit", "status")):
        print(_k, str(_v)[:1700])

print("=== FIRST STAY: STATE AND PREDICTION ALIGNMENT ===")
_r39_id = int(head_meta["stay_id"].iloc[0])
_r39_si = np.flatnonzero(states0["stay_id"] == _r39_id)
print(pd.DataFrame({
    "state_idx": _r39_si,
    "hour": states0["hour"][_r39_si],
    "insulin": states0["has_insulin"][_r39_si],
    "norepinephrine": states0["has_norepinephrine"][_r39_si]
}).head(15).to_string(index=False))
print(head_meta.loc[
    head_meta["stay_id"].eq(_r39_id),
    ["hour", "event", "state_idx", "dt"]
].head(16).to_string(index=False))

repair_read_states = np.asarray(
    states0["x"][head_meta["state_idx"].to_numpy(dtype=int)],
    dtype=np.float64
)
repair_state_blocks = {
    "out": repair_read_states[:, :192],
    "mean_h": repair_read_states[:, 192:960].reshape(-1, 4, 192).mean(1),
    "posterior_z": repair_read_states[:, 960:],
    "all_h": repair_read_states[:, 192:960]
}
repair_state_decode_results = {}
print("=== ACTIVE-DOSE DECODABILITY BY STATE BLOCK ===")
for _drug, _event in (
    ("insulin", "glucose_finger"),
    ("norepinephrine", "map_arterial")
):
    _matched = repair_aligned[_drug]
    _ok = (_matched["rd"].notna() &
           head_meta["event"].eq(_event)).to_numpy()
    _target = _matched.loc[_ok, "rd"].to_numpy()
    _folds = repair_stay_folds[_ok]
    repair_state_decode_results[_drug] = []
    for _block_name, _raw in repair_state_blocks.items():
        _f = _raw[_ok]
        _f = (_f - _f.mean(0)) / np.maximum(_f.std(0), 0.03)
        for _alpha in (10., 1000.):
            _pred = np.zeros(len(_target))
            for _fold in range(5):
                _tr = _folds != _fold
                _te = ~_tr
                _fit = Ridge(alpha=_alpha).fit(_f[_tr], _target[_tr])
                _pred[_te] = _fit.predict(_f[_te])
            _result = {
                "block": _block_name, "alpha": _alpha,
                "R2": float(r2_score(_target, _pred)),
                "corr": float(np.corrcoef(_target, _pred)[0, 1])
            }
            repair_state_decode_results[_drug].append(_result)
            print(_drug, _result)

with open("/work/notes.md", "a") as _f:
    _f.write(
        "\n## Run 38 result / Run 39\n"
        "Confirmed DML outcomes are next-reading minus last-reading, not "
        "levels. Head-feature innovation decoders failed cross-stay "
        "validation (best all-row R2 insulin -0.003, NE -0.036). "
        "Rejecting that head-only edit. Testing earlier state/posterior "
        "signals and recovering the explicit action pathway instead. "
        "No current edits.\n"
    )
```