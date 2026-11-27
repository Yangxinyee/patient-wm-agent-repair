The run stopped at a NumPy/PyTorch type mismatch before producing diagnostics; no weights changed. I will repeat the forward-only head calculation using PyTorch consistently. Close agreement with the forward tool, followed by reliable held-out dose decoding, would support a readout repair. Otherwise I will investigate the action pathway.

```python
import torch
import torch.nn.functional as F
from scipy.special import softmax
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

torch.set_grad_enabled(False)
torch.set_num_threads(8)
head_W_repair = {n: torch.as_tensor(load(n), dtype=torch.float32) for n in NAMES}

def head_linear_fixed(x, prefix):
    return F.linear(x, head_W_repair[prefix + ".weight"],
                    head_W_repair.get(prefix + ".bias"))

def head_norm_fixed(x, prefix):
    return F.layer_norm(x, (x.shape[-1],),
                        head_W_repair[prefix + ".weight"],
                        head_W_repair[prefix + ".bias"], eps=1e-5)

def head_mlp_fixed(x, prefix):
    return head_linear_fixed(F.gelu(head_linear_fixed(x, prefix + ".0")),
                             prefix + ".2")

_sx_t = torch.as_tensor(np.asarray(state_o["x"])[fact_prev_repair],
                       dtype=torch.float32)
_dt_t = torch.as_tensor(fact_dt_repair, dtype=torch.float32)
_hflow_t = _sx_t[:, 192:960].reshape(-1, 4, 192) * torch.exp(
    -_dt_t[:, None, None] * torch.exp(head_W_repair["core.flow.a"]))
_context_t = (_sx_t[:, :192]
              + head_norm_fixed(_hflow_t.mean(dim=1), "core.out_norm")
              + head_mlp_fixed(torch.log1p(_dt_t)[:, None] / 10.0, "gap"))
_prior_t = head_mlp_fixed(_context_t, "z_prior")
print("prior shape:", tuple(_prior_t.shape),
      "log-SD range:", _prior_t[:, 32:].min().item(), _prior_t[:, 32:].max().item())

_lf_t = torch.zeros((len(baseline_fact), 8), dtype=torch.float32)
_lf_t[:, 0] = torch.as_tensor(fact_has_repair, dtype=torch.float32)
_lf_t[:, 3] = torch.as_tensor(fact_age_repair, dtype=torch.float32)
_lf_t[:, 6] = torch.as_tensor(fact_last_z_repair, dtype=torch.float32)
_codes_t = torch.as_tensor(fact_code_repair, dtype=torch.long)
_bins_t = torch.as_tensor(fact_bin_repair, dtype=torch.long)
_register_t = (
    head_W_repair["embed.code.weight"][_codes_t]
    + head_linear_fixed(_lf_t, "last_numeric")
    + head_W_repair["embed.bins.value.weight"][_bins_t]
    + head_W_repair["embed.bins.rate.weight"][0]
)
_transition_t = head_W_repair["embed_transition.value_bin.weight"][
    _codes_t * 17 + _bins_t]
_g = torch.Generator().manual_seed(2401)
_phi_sum_t = torch.zeros((len(_lf_t), 192))
_prob_sum_t = torch.zeros((len(_lf_t), 17))
for _draw in range(16):
    _zp_t = _prior_t[:, :32] + torch.exp(_prior_t[:, 32:].clamp(-10, 5)) * \
            torch.randn((len(_lf_t), 32), generator=_g)
    _view_t = _context_t + head_linear_fixed(_zp_t, "z_head")
    _q_t = head_norm_fixed(_view_t + head_W_repair["type_emb.weight"][1],
                          "query_norm")
    _wc_t = _q_t + _register_t
    _phi_t = F.gelu(head_linear_fixed(_wc_t, "attr_heads.value_bin.0"))
    _logits_t = head_linear_fixed(_phi_t, "attr_heads.value_bin.2") + _transition_t
    _phi_sum_t += _phi_t
    _prob_sum_t += torch.softmax(_logits_t, dim=1)

head_phi_repair = (_phi_sum_t / 16).numpy()
head_prob_repair = (_prob_sum_t / 16).numpy()
_head_mean = (head_prob_repair * fact_centres_repair).sum(axis=1)
_head_nll = -np.log(np.maximum(head_prob_repair[
    np.arange(len(_lf_t)), fact_observed_bin_repair], 1e-15))
_head_err = {
    "mean_RMSE": float(np.sqrt(np.mean((_head_mean-reference_mean_repair)**2))),
    "nll_RMSE": float(np.sqrt(np.mean((_head_nll-reference_nll_repair)**2))),
    "mean_NLL": float(_head_nll.mean())
}
print("CPU head check:", {k: round(v, 5) for k, v in _head_err.items()})

print("\n=== INNOVATION DECODING, LAST-BIN CENTRED ===")
head_innovation_probes = {}
_bf = baseline_fact.reset_index(drop=True)
_t_lookup = {idx: i for i, idx in enumerate(tpan.index)}
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
    _feat = head_phi_repair[_rows].astype(float)
    _bm = np.zeros((17, _feat.shape[1]))
    _ym = np.zeros(17)
    for _b in range(17):
        _sel = _fit & (_bin == _b)
        if _sel.any():
            _bm[_b] = _feat[_sel].mean(axis=0)
            _ym[_b] = _eps[_sel].mean()
    _fc = _feat - _bm[_bin]
    _yc = _eps - _ym[_bin]
    _scale = np.maximum(_fc[_fit].std(axis=0), 0.03)
    _Z = _fc / _scale
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
print("\nIndependent training-state IDs:", repair_train_ids)
with open("/work/notes.md", "a") as _f:
    _f.write("\n## Runs 23–24\n"
             "Run 23 failed before diagnostics due to NumPy/Tensor mixing. Run 24 used "
             "forward-only PyTorch to reconstruct value-head features and test stay-separated "
             "dose-innovation decoding. Head check: " + str(_head_err) +
             ". No weights edited.\n")
```