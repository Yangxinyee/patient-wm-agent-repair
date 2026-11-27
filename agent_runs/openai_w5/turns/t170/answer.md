The zero insulin slope came from a sign error in my constrained fit, not evidence of an absent effect. The MAP tilt also responds much more strongly on cuff than arterial readings. I’ll fix the sign and test separate sensor calibrations, rejecting any candidate whose factual cost is excessive.

```python
from scipy.optimize import nnls
from scipy.special import softmax, ndtr
from sklearn.linear_model import Ridge

_r148_reserved = np.argsort(_r141_energy)[:8]
_r148_keep = np.setdiff1d(np.arange(192), _r148_reserved)
_r148_H = np.asarray(_r141_H, dtype=np.float64)
_r148_flat = _r148_H.reshape(-1, 192)
_r148_mask = np.tile(_r147_fit, _r148_H.shape[0])
_r148_lost = _r148_flat[:, _r148_reserved] @ _r141_W2[:, _r148_reserved].T
_r148_distill = Ridge(alpha=.001).fit(
    _r148_flat[_r148_mask][:, _r148_keep], _r148_lost[_r148_mask])
_r148_error = (
    _r148_distill.predict(_r148_flat[:, _r148_keep]) - _r148_lost
).reshape(_r148_H.shape[0], len(_r140_df), 17)
_r148_base = np.asarray(_r140_features["logits"], dtype=np.float64) + _r148_error
_r148_Pmc = softmax(_r148_base, axis=-1)
_r148_mu = np.sum(_r148_Pmc * _r141_centers[None], axis=-1)
_r148_J = np.mean(
    _r148_Pmc * (_r141_centers[None] - _r148_mu[..., None]), axis=0)

_r148_events = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
_r148_event_array = _r140_df.event.to_numpy()
_r148_specs = []
_r148_components = []
print("Reserved neurons:", _r148_reserved)
for event, j, beta, c in [
    ("glucose_finger", 0, -1.83815, .10),
    ("glucose_lab", 0, -1.83815, .10),
    ("map_arterial", 1, 3.6, .40),
    ("map_cuff", 1, 3.6, .40),
]:
    own = _r148_event_array == event
    decoded = _r140_pair[:, j] * _r140_scale[j]
    proxy = Ridge(alpha=.1).fit(
        _r147_X[_r147_fit & own], decoded[_r147_fit & own])
    gate = Ridge(alpha=.1).fit(
        _r147_X[_r147_fit], np.where(own[_r147_fit], 1., -1.))
    score = gate.predict(_r147_X)
    margin = np.where(own, score, -score)
    m = _r147_fit & own & (_r147_dose[:, j] > 0)
    transport_ratio = np.clip(
        decoded[m] / _r147_dose[m, j], .25, 3.)
    A = (_r148_J[m] @ _r147_basis) * transport_ratio[:, None]
    w = np.sqrt(_r147_dose[m, j] / np.mean(_r147_dose[m, j]))
    meanrow = np.average(A, axis=0, weights=_r147_dose[m, j])
    smooth = np.diff(np.eye(15), axis=0)
    system = np.vstack([
        A*w[:, None], 5*np.sqrt(len(A))*meanrow[None],
        .5*np.sqrt(len(A))*smooth, .05*np.sqrt(len(A))*np.eye(15)])
    rhs = np.r_[np.full(len(A), abs(beta))*w,
                5*np.sqrt(len(A))*abs(beta), np.zeros(29)]
    coef, _ = nnls(system, rhs)
    tilt = np.sign(beta) * (_r147_basis @ coef)
    tilt -= tilt[1:].mean()
    dose_vector = np.zeros(192)
    dose_vector[R117[2*j]] = _r140_scale[j]
    dose_vector[R117[2*j+1]] = -_r140_scale[j]
    gvec = 8*gate.coef_
    gbias = 8*gate.intercept_
    spec = dict(event=event, drug=j, beta=beta, c=c, proxy=proxy, gate=gate,
                tilt=tilt, row_signal=gvec+c*(dose_vector-proxy.coef_),
                bias_signal=gbias-c*proxy.intercept_,
                row_control=gvec, bias_control=gbias, out=tilt/c)
    _r148_specs.append(spec)
    signal = _r147_wc @ spec["row_signal"] + spec["bias_signal"]
    control = _r147_wc @ spec["row_control"] + spec["bias_control"]
    diff = signal*ndtr(signal)-control*ndtr(control)
    component = diff[..., None]*spec["out"]
    _r148_components.append(component)
    held = _r147_test & own & (_r147_dose[:, j] > 0)
    ratio = np.clip(decoded[held]/_r147_dose[held, j], .25, 3.)
    slopes = (_r148_J[held] @ tilt)*ratio
    residual = decoded - proxy.predict(_r147_X)
    print(event,
          "gate minimum held-out margin", round(margin[_r147_test].min(), 4),
          "innovation RMSE", round(np.sqrt(np.mean(residual[_r147_test & own]**2)), 4),
          "held-out physical slope mean/std", round(slopes.mean(), 4), round(slopes.std(), 4))

_r148_I = _r148_components[0] + _r148_components[1]
_r148_NE = _r148_components[2] + _r148_components[3]
print("=== OFFLINE FACTUAL GRID, CHANGE FROM CLEAN BASE ===")
for aI, aNE in [(0, 0), (.5, .5), (1, .5), (.5, 1), (1, 1)]:
    P = softmax(_r148_base+aI*_r148_I+aNE*_r148_NE, axis=-1).mean(axis=0)
    nll = -np.log(P[np.arange(len(P)), _r141_bins].clip(1e-12))
    delta = nll-_r141_baseNLL
    print("I/NE", aI, aNE,
          "all", round(delta.mean(), 6),
          "held-out", round(delta[_r147_test].mean(), 6),
          "by sensor", {
              e: round(delta[_r147_test & (_r148_event_array == e)].mean(), 6)
              for e in _r148_events})
print("=== SAVED EFFECT TARGETS ===")
print(target_checks_A)
print(notes_text[3200:6100])
print("No weights changed.")
```
