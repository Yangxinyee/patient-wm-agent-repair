Insulin innovations are decodable from the recurrent state, especially the posterior latent, but norepinephrine is much weaker. This suggests the reading head may discard useful action information. I’ll separate active-treatment decoding from the earlier all-event test; if decoding works within each reading type, a code-gated head repair remains possible.

```python
repair_active_head_tests = {}
_r40_q = qj.detach().cpu().numpy().astype(np.float64)

_r40_codes = torch.tensor(
    [code_ids[e] for e in head_meta["event"]], dtype=torch.long
)
_r40_nf = torch.zeros((len(head_meta), 8), dtype=torch.float32)
_r40_has = torch.tensor(head_meta["has_last"].to_numpy(), dtype=torch.float32)
_r40_nf[:, 0] = _r40_has
_r40_nf[:, 1] = torch.log1p(torch.tensor(
    head_meta["age"].to_numpy() / 3600., dtype=torch.float32
)) / 10.
_r40_nf[:, 3] = _r40_has
_r40_nf[:, 6] = torch.tensor(head_meta["last_z"].to_numpy(), dtype=torch.float32)
_r40_lb = torch.tensor(head_meta["last_bin"].to_numpy(), dtype=torch.long)
_r40_fixed = (
    W0["embed.code.weight"][_r40_codes]
    + rc_lin(_r40_nf, "last_numeric")
    + W0["embed.bins.value.weight"][_r40_lb]
    + W0["embed.bins.rate.weight"][0]
    + W0["last_status.weight"][0]
)
repair_wc_mean = _r40_q + _r40_fixed.numpy()
_r40_blocks = {
    "head_hidden_mean": repair_phi,
    "query_mean": _r40_q,
    "with_code_mean": repair_wc_mean
}

print("=== ACTIVE-ONLY HEAD DECODING ===")
for _drug, _event in (
    ("insulin", "glucose_finger"),
    ("norepinephrine", "map_arterial")
):
    _matched = repair_aligned[_drug]
    _ok = (_matched["rd"].notna() &
           head_meta["event"].eq(_event)).to_numpy()
    _target = _matched.loc[_ok, "rd"].to_numpy()
    _folds = repair_stay_folds[_ok]
    repair_active_head_tests[_drug] = []
    for _name, _raw in _r40_blocks.items():
        _f = _raw[_ok]
        _f = (_f - _f.mean(0)) / np.maximum(_f.std(0), .03)
        for _alpha in (1., 10., 100.):
            _pred = np.zeros(len(_target))
            for _fold in range(5):
                _tr = _folds != _fold
                _te = ~_tr
                _fit = Ridge(alpha=_alpha).fit(_f[_tr], _target[_tr])
                _pred[_te] = _fit.predict(_f[_te])
            _result = {
                "features": _name, "alpha": _alpha,
                "R2": float(r2_score(_target, _pred)),
                "corr": float(np.corrcoef(_target, _pred)[0, 1])
            }
            repair_active_head_tests[_drug].append(_result)
            print(_drug, _result)

print("=== RECONSTRUCTION FUNCTIONS USED BY CACHE ===")
for _name in ("fn", "actfn", "rc_act", "rh_register"):
    _fn = globals().get(_name)
    if callable(_fn) and hasattr(_fn, "__code__"):
        print(_name, "constants:", _fn.__code__.co_consts)
print("Query norms:", np.quantile(np.linalg.norm(_r40_q, axis=1), [0, .5, 1]))
print("Hidden norms:", np.quantile(np.linalg.norm(repair_phi, axis=1), [0, .5, 1]))

with open("/work/notes.md", "a") as _f:
    _f.write(
        "\n## Run 39 result / Run 40\n"
        "Insulin dose innovation has held-out R2 about 0.27 in out/mean_h, "
        "0.40 in posterior z, and 0.37 in all slots (active primary rows). "
        "NE innovation remains weak (posterior z R2 0.058; out negative). "
        "Testing whether the earlier head decoder failed because it "
        "required the same linear readout to be zero for other codes.\n"
    )
```