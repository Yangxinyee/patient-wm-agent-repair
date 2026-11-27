The head-input signal recovers insulin closely, and sensor-family separation is perfect on held-out stays. Norepinephrine recovery is noisier. I’ll free four value-head neurons by regression distillation and fit centered dose signals. Small held-out distillation loss and stable dose centering would support the next edit.

```python
from scipy.special import softmax as _r141_softmax
from sklearn.linear_model import Ridge as _r141_Ridge

_r141_H = np.asarray(_r140_features["hidden"]).astype(np.float64)
_r141_logits = np.asarray(_r140_features["logits"]).astype(np.float64)
_r141_W2 = np.asarray(_r140_base_weights["attr_heads.value_bin.2.weight"]).astype(np.float64)
_r141_b2 = np.asarray(_r140_base_weights["attr_heads.value_bin.2.bias"]).astype(np.float64)
_r141_centers = np.asarray(_r140_cache["centers"])
_r141_bins = np.asarray(_r140_cache["targetbin"], dtype=int)
_r141_baseP = _r141_softmax(_r141_logits, axis=-1).mean(0)
_r141_baseNLL = -np.log(np.maximum(
    _r141_baseP[np.arange(len(_r141_bins)), _r141_bins], 1e-12
))

_r141_energy = _r141_H.var(axis=(0, 1)) * (
    (_r141_W2 - _r141_W2.mean(0, keepdims=True))**2
).sum(0)
_r141_reserved = np.argsort(_r141_energy)[:4]
_r141_keep = np.setdiff1d(np.arange(192), _r141_reserved)
_r141_flat = _r141_H.reshape(-1, 192)
_r141_mask = np.tile(_r140_fitmask, _r141_H.shape[0])
_r141_dropped = _r141_flat[:, _r141_reserved] @ _r141_W2[:, _r141_reserved].T
_r141_distill = []

print("=== FOUR-NEURON CAPACITY DISTILLATION ===")
print("Reserved neurons:", _r141_reserved.tolist())
for _r141_alpha in [0.001, 0.01, 0.1, 1.0]:
    _r141_reg = _r141_Ridge(alpha=_r141_alpha).fit(
        _r141_flat[_r141_mask][:, _r141_keep], _r141_dropped[_r141_mask]
    )
    _r141_error = (
        _r141_reg.predict(_r141_flat[:, _r141_keep]) - _r141_dropped
    ).reshape(_r141_logits.shape)
    _r141_newlogits = _r141_logits + _r141_error
    _r141_P = _r141_softmax(_r141_newlogits, axis=-1).mean(0)
    _r141_nll = -np.log(np.maximum(
        _r141_P[np.arange(len(_r141_bins)), _r141_bins], 1e-12
    ))
    _r141_kl = np.sum(_r141_baseP * np.log(
        np.maximum(_r141_baseP, 1e-12) / np.maximum(_r141_P, 1e-12)
    ), axis=1)
    print("alpha", _r141_alpha,
          "held-out KL", round(float(_r141_kl[_r140_testmask].mean()), 8),
          "held-out NLL change",
          round(float((_r141_nll - _r141_baseNLL)[_r140_testmask].mean()), 7))
    _r141_distill.append({
        "alpha": _r141_alpha, "reg": _r141_reg,
        "kl": float(_r141_kl[_r140_testmask].mean()),
        "logits": _r141_newlogits
    })
_r141_best = min(_r141_distill, key=lambda d: d["kl"])

print("=== CENTERED DOSE-PROXY REGRESSIONS ===")
_r141_nuisances = []
_r141_proxy = _r140_pair * _r140_scale
for _r141_j, _r141_family in [(0, _r140_glucose), (1, ~_r140_glucose)]:
    _r141_fit = _r140_fitmask & _r141_family
    _r141_test = _r140_testmask & _r141_family
    _r141_choices = []
    for _r141_alpha in [0.1, 1.0, 10.0, 100.0]:
        _r141_reg = _r141_Ridge(alpha=_r141_alpha).fit(
            _r140_X[_r141_fit], _r141_proxy[_r141_fit, _r141_j]
        )
        _r141_pred = _r141_reg.predict(_r140_X)
        _r141_rmse = float(np.sqrt(np.mean(
            (_r141_proxy[_r141_test, _r141_j] - _r141_pred[_r141_test])**2
        )))
        _r141_choices.append((_r141_rmse, _r141_alpha, _r141_reg))
    _r141_choice = min(_r141_choices, key=lambda x: x[0])
    _r141_nuisances.append(_r141_choice[2])
    print(["insulin", "norepinephrine"][_r141_j],
          "held-out RMSE by alpha",
          [(a, round(e, 5)) for e, a, r in _r141_choices],
          "selected", _r141_choice[1])

print("=== RECORD-TARGETED LOGIT-TILT FITS ===")
_r141_Pmc = _r141_softmax(_r141_best["logits"], axis=-1)
_r141_mumc = (_r141_Pmc * _r141_centers[None, :, :]).sum(-1)
_r141_J = (
    _r141_Pmc * (_r141_centers[None, :, :] - _r141_mumc[:, :, None])
).mean(0)
_r141_curvature = np.diff(np.eye(16), n=2, axis=0)
_r141_tilts = []
for _r141_j, _r141_family, _r141_beta in [
    (0, _r140_glucose, -1.838), (1, ~_r140_glucose, 3.60)
]:
    _r141_A = _r141_J[_r140_fitmask & _r141_family, 1:]
    _r141_penalty = (
        _r141_curvature.T @ _r141_curvature + 0.01 * np.eye(16)
    )
    _r141_L = np.r_[0.0, np.linalg.solve(
        _r141_A.T @ _r141_A + _r141_penalty,
        _r141_A.T @ np.full(len(_r141_A), _r141_beta)
    )]
    _r141_L -= _r141_L.mean()
    _r141_tilts.append(_r141_L)
    for _r141_event in (
        ["glucose_finger", "glucose_lab"] if _r141_j == 0
        else ["map_arterial", "map_cuff"]
    ):
        _r141_m = _r140_testmask & (_r140_df.event.to_numpy() == _r141_event)
        _r141_slopes = _r141_J[_r141_m] @ _r141_L
        print(_r141_event, "held-out mean/std local slope",
              round(float(_r141_slopes.mean()), 4),
              round(float(_r141_slopes.std()), 4))
print("No weights changed in this run.")
```
