The calibration uses 100 training stays with a stay-level holdout. Sensor separation looks clean, but the fitted logit tilts are irregular enough to risk wrong-sign responses. I’ll test monotone tilts and an analytic four-neuron correction offline first. It should preserve factual likelihood while producing consistently signed dose effects.

```python
from scipy.optimize import nnls
from scipy.special import softmax
from scipy.special import ndtr
from sklearn.linear_model import Ridge

_r147_clean_weights = {n: load(n).clone() for n in NAMES}
print("Installed head matches calibration:",
      torch.equal(_r147_clean_weights["attr_heads.value_bin.0.weight"],
                  _r140_base_weights["attr_heads.value_bin.0.weight"]))
print("Ordinary-feature differences:", np.max(np.abs(_r140_X - _r140_wc), axis=0)[R117])
print("Private coefficients in proxies:",
      [r.coef_[R117].tolist() for r in _r141_nuisances])
print("Private coefficients in sensor gate:", _r140_gate.coef_[R117])

_r147_wc = np.asarray(_r140_wc_all, dtype=np.float64)
_r147_X = np.asarray(_r140_wc, dtype=np.float64).copy()
_r147_X[:, R117] = 0
_r147_fit = np.asarray(_r140_fitmask, dtype=bool)
_r147_test = np.asarray(_r140_testmask, dtype=bool)
_r147_family = np.asarray(_r140_glucose, dtype=bool)
_r147_dose = np.asarray(_r140_doses, dtype=np.float64)
_r147_proxy_models = []
for j in range(2):
    fam = _r147_family if j == 0 else ~_r147_family
    rr = Ridge(alpha=0.1).fit(
        _r147_X[_r147_fit & fam], _r147_dose[_r147_fit & fam, j])
    _r147_proxy_models.append(rr)
    decoded = _r140_pair[:, j] * _r140_scale[j]
    m = _r147_test & fam & (_r140_df.hour.to_numpy() > 0)
    print(["insulin", "NE"][j],
          "transport RMSE", np.sqrt(np.mean((decoded[m] - _r147_dose[m, j])**2)),
          "proxy RMSE", np.sqrt(np.mean((rr.predict(_r147_X[m]) - _r147_dose[m, j])**2)))

_r147_gate = Ridge(alpha=1).fit(
    _r147_X[_r147_fit], np.where(_r147_family[_r147_fit], 1., -1.))
_r147_distilled = np.asarray(_r141_best["logits"], dtype=np.float64)
_r147_Pmc = softmax(_r147_distilled, axis=-1)
_r147_mu = np.sum(_r147_Pmc * _r141_centers[None], axis=-1)
_r147_J = np.mean(
    _r147_Pmc * (_r141_centers[None] - _r147_mu[..., None]), axis=0)
_r147_basis = (
    np.arange(17)[:, None] >= np.arange(2, 17)[None, :]).astype(float)
_r147_tilts = []
for j, beta in enumerate((-1.83815, 3.6)):
    fam = _r147_family if j == 0 else ~_r147_family
    m = _r147_fit & fam & (_r147_dose[:, j] > 0)
    A = np.sign(beta) * (_r147_J[m] @ _r147_basis)
    w = np.sqrt(_r147_dose[m, j] / np.mean(_r147_dose[m, j]))
    aw = A * w[:, None]
    bw = np.full(len(A), abs(beta)) * w
    meanrow = np.average(A, axis=0, weights=_r147_dose[m, j])
    smooth = np.diff(np.eye(15), axis=0)
    system = np.vstack([
        aw, 5*np.sqrt(len(A))*meanrow[None],
        0.5*np.sqrt(len(A))*smooth,
        0.05*np.sqrt(len(A))*np.eye(15)])
    rhs = np.r_[bw, 5*np.sqrt(len(A))*abs(beta),
                np.zeros(smooth.shape[0]+15)]
    coef, _ = nnls(system, rhs)
    tilt = np.sign(beta) * (_r147_basis @ coef)
    tilt -= tilt[1:].mean()
    _r147_tilts.append(tilt)
    for event in (["glucose_finger", "glucose_lab"] if j == 0
                  else ["map_arterial", "map_cuff"]):
        v = _r147_test & (_r140_df.event.to_numpy() == event) & (_r147_dose[:, j] > 0)
        s = _r147_J[v] @ tilt
        print(event, "held-out slope mean/std", round(s.mean(), 4), round(s.std(), 4))

_r147_neurons = []
for j, channel_scale in enumerate((0.10, 0.40)):
    reg = _r147_proxy_models[j]
    dose_vector = np.zeros(192)
    dose_vector[R117[2*j]] = _r140_scale[j]
    dose_vector[R117[2*j+1]] = -_r140_scale[j]
    residual_vector = dose_vector - reg.coef_
    gate_sign = 1 if j == 0 else -1
    gate_vector = 8 * gate_sign * _r147_gate.coef_
    gate_bias = 8 * gate_sign * _r147_gate.intercept_
    _r147_neurons.append({
        "row_signal": gate_vector + channel_scale*residual_vector,
        "bias_signal": gate_bias - channel_scale*reg.intercept_,
        "row_control": gate_vector,
        "bias_control": gate_bias,
        "out": _r147_tilts[j]/channel_scale,
    })

def _r147_gelu(x):
    return x * ndtr(x)

_r147_correction = np.zeros_like(_r147_distilled)
for spec in _r147_neurons:
    signal = _r147_wc @ spec["row_signal"] + spec["bias_signal"]
    control = _r147_wc @ spec["row_control"] + spec["bias_control"]
    diff = _r147_gelu(signal) - _r147_gelu(control)
    _r147_correction += diff[..., None] * spec["out"]
    print("Gated innovation quantiles:", np.quantile(diff, [.01, .5, .99]))

print("=== OFFLINE FACTUAL GRID: RELATIVE TO CLEAN CHANNEL BASE ===")
for strength in (0., .25, .5, .75, 1.):
    P = softmax(_r147_distilled + strength*_r147_correction, axis=-1).mean(axis=0)
    nll = -np.log(P[np.arange(len(P)), _r141_bins].clip(1e-12))
    print(strength, "held-out NLL change",
          round(np.mean((nll-_r141_baseNLL)[_r147_test]), 6),
          "all-row change", round(np.mean(nll-_r141_baseNLL), 6))
print("No weights changed.")
```
