The history-adjusted estimates are about −2.05 glucose units per insulin rate unit and +3.17 MAP units per norepinephrine rate unit, much larger than the original rollout responses. Agreement with the insulin threshold estimates strengthens—but does not prove—the causal interpretation. I will separate the later-dose feedback from the impulse responses and translate the resulting kernels into provisional rollout targets. Stable kernels under better history adjustment would confirm them; substantial changes would refute them.

```python
print("=== REMAINING RETAINED OBJECTS ===")
print("\n".join(_items)[4200:8200])

print("\n=== NUISANCE MODEL STRUCTURE ===")
for _drug in ["insulin", "norepinephrine"]:
    _m = dml_models[_drug]
    print(_drug, "container:", type(_m).__name__, "length:", len(_m))
    for _j, _part in enumerate(_m[:2]):
        if isinstance(_part, (list, tuple)):
            print(" fold", _j, [(type(x).__name__, getattr(x, "n_features_in_", None)) for x in _part])
        elif isinstance(_part, dict):
            print(" fold", _j, list(_part.keys()))
        else:
            print(" fold", _j, type(_part).__name__)

print("\n=== PROVISIONAL FLEXIBLE IMPULSE KERNELS ===")
# These numbers are the record-derived moments printed by run 19.
# Deconvolution is provisional: horizon-specific selection can violate
# the assumption that the same homogeneous impulse kernel applies.
_moment_inputs = {
    "insulin": {
        "response": [-2.07327, -2.38422, -1.94979, -1.41475, -0.89005, -0.52098],
        "cov": [[1], [1, -.04674], [1, -.04763, -.05417],
                [1, -.04950, -.05599, -.04639],
                [1, -.05374, -.05971, -.05135, -.03550],
                [1, -.05594, -.06197, -.05474, -.03721, -.01631]]
    },
    "norepinephrine": {
        "response": [3.20132, 4.98967, 5.30243, 4.96460, 4.37750, 3.90066],
        "cov": [[1], [1, .74903], [1, .75240, .58721],
                [1, .75571, .59314, .47901],
                [1, .75628, .59468, .47853, .38594],
                [1, .76084, .59768, .48145, .38742, .31589]]
    }
}
provisional_kernels = {}
for _drug, _v in _moment_inputs.items():
    _g = []
    for _h, _r in enumerate(_v["response"]):
        _feedback = sum(_v["cov"][_h][_j] * _g[_h-_j]
                        for _j in range(1, _h+1))
        _g.append(_r - _feedback)
    provisional_kernels[_drug] = np.array(_g)
    print(_drug, "lags 1–6:", np.round(_g, 4).tolist())

print("\n=== PROVISIONAL TARGETS ON ORIGINAL FORWARD'S ELIGIBLE HOURS ===")
for _drug, _edit_name in [("insulin", "insulin_x2"),
                           ("norepinephrine", "ne_x2")]:
    _rr = base_roll[(base_roll["edit"] == _edit_name)
                    & base_roll["reached"].astype(str).str.lower().isin(["true", "1"])].copy()
    _target_rows = []
    for _row in _rr.itertuples(index=False):
        _sid, _hour = int(_row.stay_id), int(_row.hour)
        _effect = 0.0
        for _lag, _g in enumerate(provisional_kernels[_drug], start=1):
            _t = _hour - _lag
            if _t >= 12 and (_sid, _t) in tpan.index:
                _rate = tpan.loc[(_sid, _t), _drug]
                _effect += _g * (float(_rate) if pd.notna(_rate) else 0.0)
        _target_rows.append((_sid, _hour, _effect))
    _targets = pd.DataFrame(_target_rows, columns=["stay_id", "hour", "effect"])
    print(_drug, "6-lag truncation:",
          "hour-weighted", round(_targets["effect"].mean(), 4),
          "stay-weighted", round(_targets.groupby("stay_id")["effect"].mean().mean(), 4),
          "half-dose is minus one-half only under linearity")

print("\n=== HEAD HELPER CONFIGURATION OBJECTS ===")
for _n, _o in list(globals().items()):
    if _n.startswith("_"):
        continue
    if isinstance(_o, dict) and any(_k in _o for _k in
            ["gap_activation", "head_activation", "zcol", "agecol"]):
        print(_n, _o)

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 22\n"
             "Recovered remaining retained objects. Deconvolved run-19 closed-loop dose-innovation "
             "moments into a provisional six-lag direct-response kernel and translated it to "
             "the tuning forward's eligible hours. This is a diagnostic, not accepted ground truth: "
             "horizon selection, nuisance misspecification and response heterogeneity remain "
             "possible. No weights edited.\n")
```