The standard type, activation, and gap choices fit best, but the remaining error is too large to attribute to sampling alone. The last-status embedding is zero here, so it was not the cause. I will inspect the register inputs and check drug-effect estimates across dose regimes. Stable slopes would support a shared correction; substantial variation would argue against a simple linear edit.

```python
print("=== REGISTER DIAGNOSTICS ===")
print("last_numeric column norms:",
      torch.linalg.vector_norm(W0["last_numeric.weight"], dim=0).tolist())
print("head_mc_layout variables:", head_mc_layout.__code__.co_varnames)
print("reading_cache variables:", reading_cache.__code__.co_varnames)
print("numeric metadata:", norm_meta)
_diag = cacheA["df"].head(12)[["stay_id", "hour", "event", "value"]].copy()
for _k in ("dt", "haslast", "lastval", "lastage", "lastbin", "targetbin"):
    _diag[_k] = torch.as_tensor(cacheA[_k])[:12].reshape(-1).numpy()
_diag["gpu_mean"] = np.asarray(cacheA["gpu_mean"])[:12]
print(_diag.to_string(index=False))

print("\n=== EFFECT-ANALYSIS OBJECTS ===")
print("testpanel index:", testpanel.index.names, "first:", testpanel.index[:4].tolist())
print("residual shapes:", {k: np.asarray(v).shape for k, v in residuals.items()})
for _k, _m in nuisance_models.items():
    _params = _m.get_params()
    print(_k, type(_m).__name__, {
        k: _params[k] for k in
        ("max_iter", "max_leaf_nodes", "max_depth", "l2_regularization",
         "learning_rate", "min_samples_leaf") if k in _params
    })
print("dose support:")
for _drug in ("insulin", "norepinephrine"):
    _v = panel[_drug].fillna(0)
    print(_drug, _v.quantile([0, .25, .5, .75, .9, .99, 1]).to_dict(),
          "positive fraction", float((_v > 0).mean()))

if all(len(np.asarray(residuals[k])) == len(testpanel)
       for k in ("insulin", "norepinephrine", "G_next", "M_next")):
    innovation_A = np.column_stack([
        np.asarray(residuals["insulin"]), np.asarray(residuals["norepinephrine"])
    ])
    innovation_groups = np.asarray(testpanel.index.get_level_values(0))
    print("\n=== HELD-OUT CONDITIONAL DOSE STRATA ===")
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
            _coef, _se = clustered_ols(
                innovation_A[_mask], _yr[_mask], innovation_groups[_mask])
            _row = (_drug, _s, int(_mask.sum()), float(_cuts[_s]),
                    float(_cuts[_s+1]), float(_coef[_j+1]), float(_se[_j+1]))
            strata_results.append(_row)
            print("drug,stratum,n,expected-dose range,beta,SE:", _row)

print("\n=== ASSUMPTION-DEPENDENT ROLLOUT TARGETS ===")
print("mechanistic function constants:", mechanistic_shifts.__code__.co_consts)
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
            else:
                print(_drug, _mult, type(_res).__name__, str(_res)[:350])
        except Exception as _exc:
            print(_drug, _mult, type(_exc).__name__, str(_exc))

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 17\n"
             "- Type=1, GELU and log-gap/10 remain the best documented head conventions.\n"
             "- The absent last-status embedding is exactly zero; it cannot explain "
             "the reconstruction discrepancy.\n"
             "- Inspecting register features, dose support and conditional-dose strata "
             "before choosing an edit or treating linear dynamic targets as established.\n")
```
