Insulin doses span a broad range, while norepinephrine rates stop at 1, so some doubled-rate effects will require extrapolation. The stratified analysis failed before producing estimates because I unpacked its return value incorrectly. I will fix that and test bin-edge and register-flag conventions. Consistent slopes and a markedly better head match would justify moving toward an edit.

```python
print("=== CORRECTED HELD-OUT DOSE STRATA ===")
strata_results = []
for _drug, _out, _j in (("insulin", "G_next", 0),
                       ("norepinephrine", "M_next", 1)):
    _raw = testpanel[_drug].fillna(0).to_numpy()
    _expected = _raw - innovation_A[:, _j]
    _yr = np.asarray(residuals[_out])
    _finite = np.isfinite(_yr) & np.isfinite(innovation_A).all(1)
    _cuts = np.quantile(_expected[_finite], [0, 1/3, 2/3, 1])
    for _s in range(3):
        _mask = (_finite & (_expected >= _cuts[_s]) &
                 (_expected <= _cuts[_s+1] if _s == 2 else
                  _expected < _cuts[_s+1]))
        if _mask.sum() < 200:
            continue
        _fit = clustered_ols(
            innovation_A[_mask], _yr[_mask], innovation_groups[_mask])
        _coef, _se = _fit[0], _fit[1]
        _row = dict(drug=_drug, stratum=_s, n=int(_mask.sum()),
                    dose_min=float(_cuts[_s]), dose_max=float(_cuts[_s+1]),
                    beta=float(_coef[_j+1]), se=float(_se[_j+1]))
        strata_results.append(_row)
        print(_row)

print("\n=== ASSUMPTION-DEPENDENT ROLLOUT TARGETS ===")
target_ids_A = np.unique(cacheA["df"].stay_id)
target_checks_A = {}
for _drug, _out, _key in (("insulin", "glucose_finger", "G"),
                         ("norepinephrine", "map_arterial", "M")):
    for _mult in (2.0, 0.5):
        try:
            _res = mechanistic_shifts(
                panel_tune, target_ids_A, _drug, _out,
                dyn_estimates[_key]["phi"], dyn_estimates[_key]["beta"],
                multiplier=_mult)
            target_checks_A[(_drug, _mult)] = _res
            if isinstance(_res, pd.DataFrame):
                print(_drug, _mult, _res.shape,
                      _res.select_dtypes("number").mean().to_dict())
            elif isinstance(_res, dict):
                print(_drug, _mult, {
                    k: (float(np.nanmean(v)) if isinstance(v, np.ndarray)
                        else str(v)[:180]) for k, v in _res.items()
                })
            else:
                print(_drug, _mult, type(_res).__name__, str(_res)[:350])
        except Exception as _exc:
            print(_drug, _mult, type(_exc).__name__, str(_exc))

print("\n=== BIN CONVENTIONS AND REGISTER FLAGS ===")
cacheA_right = dict(cacheA)
_prev_values = cacheA["df"].groupby(["stay_id", "event"]).value.shift(1).to_numpy()
_last_right = np.zeros(len(_prev_values), dtype=np.int64)
_target_right = np.zeros(len(_prev_values), dtype=np.int64)
for _event in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff"):
    _ix = np.flatnonzero((cacheA["df"].event == _event).to_numpy())
    _edges = norm_meta[(_event, "value")]["fine_bin_edges"]
    _good = np.isfinite(_prev_values[_ix])
    _last_right[_ix[_good]] = 1 + np.searchsorted(
        _edges, _prev_values[_ix[_good]], side="right")
    _target_right[_ix] = 1 + np.searchsorted(
        _edges, cacheA["df"].value.to_numpy()[_ix], side="right")
cacheA_right["lastbin"] = torch.tensor(_last_right)
cacheA_right["targetbin"] = torch.tensor(_target_right)
print("last bins changed:", int(np.sum(
    _last_right != torch.as_tensor(cacheA["lastbin"]).numpy())))
print("target bins changed:", int(np.sum(
    _target_right != torch.as_tensor(cacheA["targetbin"]).numpy())))

_probe = np.sort(np.random.default_rng(51).choice(len(cacheA["df"]), 512, replace=False))
_gm = torch.tensor(np.asarray(cacheA["gpu_mean"])[_probe], dtype=torch.float32)
_gn = torch.tensor(np.asarray(cacheA["gpu_nll"])[_probe], dtype=torch.float32)
_sc = torch.tensor([
    norm_meta[(e, "value")]["iqr"] for e in cacheA["df"].event.iloc[_probe]
])
bin_flag_results = []
for _side, _cache in (("left", cacheA), ("right", cacheA_right)):
    for _age, _flag in itertools.product(
            ("seconds", "minutes", "hours"), (None, 3, 5, 7)):
        _pm, _pn = reconstruct_head(
            _cache, _probe, ageunit=_age, flagcol=_flag,
            nsamples=32, **head_spec)
        _me = (((_pm - _gm) / _sc) ** 2).mean().item()
        _ne = (_pn - _gn).abs().mean().item()
        bin_flag_results.append((_me, _ne, _side, _age, str(_flag)))
bin_flag_results.sort()
for _row in bin_flag_results[:10]:
    print(_row)

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Runs 17–18\n"
             "- Run 17's strata calculation failed because clustered_ols returns "
             "more than two objects; corrected by taking its first two outputs.\n"
             f"- Conditional-dose strata: {strata_results}\n"
             "- Insulin has broad observed rate support (0–16); norepinephrine "
             "observed rates are capped at 1. Above-cap effects need a response-form assumption.\n"
             f"- Best bin/flag trial: {bin_flag_results[0]}.\n"
             "- No edits. Dynamic rollout targets remain assumption-dependent, not observed truth.\n")
```
