The provisional kernels imply much larger shifts than the model predicts: roughly −12 to −14.5 mg/dL for doubled insulin and +3.5 to +3.8 mmHg for doubled norepinephrine, before any longer-lag contribution. These remain uncertain targets. I will now test whether the value head retains usable dose information. Successful decoding would support a small readout repair; poor decoding would point toward repairing the action pathway instead.

```python
from scipy.special import softmax
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

print("=== HEAD PREFIXES AND ALIGNMENT ===")
_gap_prefixes = sorted({n[:-9] for n in NAMES
                        if "gap" in n and n.endswith(".0.weight")})
_prior_prefixes = sorted({n[:-9] for n in NAMES
                          if "prior" in n and n.endswith(".0.weight")})
_zhead_names = [n for n in NAMES
                if "z_head" in n and n.endswith(".weight")]
_type_names = [n for n in NAMES
               if "type_emb" in n and n.endswith(".weight")]
print("gap:", _gap_prefixes, "prior:", _prior_prefixes,
      "z_head:", _zhead_names, "type:", _type_names)
print("dt quantiles:", np.quantile(fact_dt_repair, [0, .5, 1]))
print("age quantiles:", np.quantile(fact_age_repair, [0, .5, 1]))
print("factual columns:", list(baseline_fact.columns))
print("policy prediction shapes:",
      {k: np.shape(v) for k, v in policy_pred_tuning.items()})

assert len(_gap_prefixes) == len(_prior_prefixes) == len(_zhead_names) == len(_type_names) == 1
_gap_prefix = _gap_prefixes[0]
_prior_prefix = _prior_prefixes[0]
_zhead_prefix = _zhead_names[0][:-7]

_sx = np.asarray(state_o["x"])[fact_prev_repair]
_hflow = _sx[:, 192:960].reshape(-1, 4, 192) * np.exp(
    -np.asarray(fact_dt_repair)[:, None, None] * np.exp(load("core.flow.a")))
_context = (_sx[:, :192]
            + norm_repair(_hflow.mean(axis=1), "core.out_norm")
            + mlp_repair(np.log1p(fact_dt_repair)[:, None] / 10.0,
                         _gap_prefix, "gelu"))
_prior = mlp_repair(_context, _prior_prefix, "gelu")
print("prior shape:", _prior.shape)

_lf = np.zeros((len(baseline_fact), 8), dtype=np.float32)
_lf[:, 0] = fact_has_repair
_lf[:, 3] = fact_age_repair
_lf[:, 6] = fact_last_z_repair
_register = (
    load("embed.code.weight")[fact_code_repair]
    + linear_repair(_lf, "last_numeric")
    + load("embed.bins.value.weight")[fact_bin_repair]
    + load("embed.bins.rate.weight")[np.zeros(len(_lf), dtype=int)]
)
_transition = load("embed_transition.value_bin.weight")[
    fact_code_repair * 17 + fact_bin_repair]
_rng_head = np.random.default_rng(2301)
_phi_sum = np.zeros((len(_lf), 192), dtype=np.float64)
_prob_sum = np.zeros((len(_lf), 17), dtype=np.float64)
# Average over prior draws; this is a forward calculation, with no gradients.
for _draw in range(16):
    _zp = _prior[:, :32] + np.exp(np.clip(_prior[:, 32:], -10, 5)) * \
          _rng_head.standard_normal((len(_lf), 32))
    _view = _context + linear_repair(_zp, _zhead_prefix)
    _q = norm_repair(_view + load(_type_names[0])[1], "query_norm")
    _wc = _q + _register
    _phi = act_repair(linear_repair(_wc, "attr_heads.value_bin.0"), "gelu")
    _logits = linear_repair(_phi, "attr_heads.value_bin.2") + _transition
    _phi_sum += _phi
    _prob_sum += softmax(_logits, axis=1)
head_phi_repair = (_phi_sum / 16).astype(np.float32)
head_prob_repair = _prob_sum / 16
_head_mean = (head_prob_repair * fact_centres_repair).sum(axis=1)
_head_nll = -np.log(np.maximum(head_prob_repair[
    np.arange(len(_lf)), fact_observed_bin_repair], 1e-15))
print("CPU versus forward mean RMSE:",
      round(float(np.sqrt(np.mean((_head_mean-reference_mean_repair)**2))), 5))
print("CPU versus forward NLL RMSE:",
      round(float(np.sqrt(np.mean((_head_nll-reference_nll_repair)**2))), 5))

print("\n=== STAY-SEPARATED INNOVATION DECODING FROM VALUE-HEAD FEATURES ===")
head_innovation_probes = {}
_bf = baseline_fact.reset_index(drop=True)
_t_index = tpan.index
_t_lookup = {idx: i for i, idx in enumerate(_t_index)}
for _drug, _event in [("insulin", "glucose_finger"),
                       ("norepinephrine", "map_arterial")]:
    _rows = np.flatnonzero(_bf["event"].to_numpy() == _event)
    _pos = np.array([_t_lookup.get((int(_bf.iloc[j]["stay_id"]),
                                  int(round(_bf.iloc[j]["hour"]))-1), -1)
                     for j in _rows])
    _good = _pos >= 0
    _rows, _pos = _rows[_good], _pos[_good]
    _dose = tpan[_drug].fillna(0).to_numpy()[_pos]
    _eps = _dose - np.asarray(policy_pred_tuning[_drug])[_pos]
    _sid = _bf.iloc[_rows]["stay_id"].to_numpy()
    _fit = (_sid % 10) < 7
    _test = ~_fit
    _bin = fact_bin_repair[_rows]
    _F = head_phi_repair[_rows].astype(float)
    # Remove a same-code last-bin intercept before probing action information.
    _bm = np.zeros((17, _F.shape[1]))
    _ym = np.zeros(17)
    for _b in range(17):
        _sel = _fit & (_bin == _b)
        if _sel.any():
            _bm[_b] = _F[_sel].mean(axis=0)
            _ym[_b] = _eps[_sel].mean()
    _Fc = _F - _bm[_bin]
    _yc = _eps - _ym[_bin]
    _scale = np.maximum(_Fc[_fit].std(axis=0), 0.03)
    _Z = _Fc / _scale
    for _alpha in [1, 10, 100]:
        _probe = Ridge(alpha=_alpha, fit_intercept=False).fit(_Z[_fit], _yc[_fit])
        _pr = _probe.predict(_Z[_test]) + _ym[_bin[_test]]
        _r2 = r2_score(_eps[_test], _pr)
        print(_drug, "alpha", _alpha, "n", len(_rows),
              "innovation SD", round(float(_eps.std()), 4),
              "test R2", round(float(_r2), 4))
        head_innovation_probes[(_drug, _alpha)] = {
            "coef": _probe.coef_ / _scale, "bin_feature_mean": _bm,
            "bin_target_mean": _ym, "rows": _rows, "eps": _eps,
            "fit": _fit, "test": _test, "r2": _r2
        }

repair_train_ids = sorted(train["stay_id"].unique().tolist())[:100]
print("\n100 training stays for an independent state sample:", repair_train_ids)

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 23\n"
             "Built prior-averaged value-head features from original tuning states and tested "
             "stay-separated dose-innovation decoding after removing last-bin intercepts. "
             "CPU reconstruction remains subject to its measured discrepancy with the forward "
             "tool. Prepared independent training-state IDs. No weights edited.\n")
```