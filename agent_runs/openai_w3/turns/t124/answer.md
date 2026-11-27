The timing is correct, but the recurrent replay still fails badly. A stale weight cache is a possible cause even though no edits are submitted. I’ll compare cached weights with the originals and recompute the posterior and prelude. That will distinguish stale-cache errors from reconstruction errors.

```python
import numpy as np, torch
import torch.nn.functional as F

print("=== CACHE AUDIT ===")
_diff113 = []
for _name113 in NAMES:
    if _name113 in W:
        _orig113 = load_original(_name113).to(dtype=torch.float32)
        _err113 = float(torch.max(torch.abs(W[_name113]-_orig113)))
        if _err113 > 1e-7:
            _diff113.append((_name113, _err113))
print("cached tensors differing from originals", _diff113)
W = {name: load_original(name).to(dtype=torch.float32) for name in NAMES}

print("\n=== FRESH POSTERIOR AND PRELUDE ===")
_outprev113 = torch.as_tensor(_x112[_prev112, :192])
_ctx113 = (_outprev113 +
           layer_norm(_hflow112.mean(dim=1), "core.out_norm") +
           mlp(torch.log(_dt112.clamp_min(1))[:, None]/10, "gap"))
_e113 = torch.as_tensor(e80[_i112], dtype=torch.float32)
_u113 = torch.as_tensor(u80[_i112], dtype=torch.float32)
_z113 = mlp(torch.cat([_ctx113, _e113, _u113], dim=1), "z_post")[:, :32]
print("posterior RMSE",
      float(np.sqrt(np.mean(
          (as_numpy(_z113)-_x112[_i112, 960:992])**2))))
_E113 = prelude81(_hflow112, _e113, _u113, _z113)
if isinstance(_E113, (tuple, list)):
    print("prelude result shapes", [tuple(v.shape) for v in _E113
                                   if hasattr(v, "shape")])
    _E113 = next(v for v in _E113 if tuple(v.shape) == tuple(Ep105.shape))
print("fresh E versus cached E RMSE",
      float(torch.sqrt(torch.mean((_E113-Ep105)**2))))

print("\n=== CORE CONVENTIONS ===")
_fit113 = []
for _current113 in (False, True):
    _mem113, _mask113, _age113 = memory82(_i112, _current113)
    for _scale113 in (1., 3600., 1./3600.):
        for _mode113 in (0, 1):
            _h113, _out113 = state_update82(
                _hflow112, _E113, _mem113, _mask113,
                _age113*_scale113, _mode113)
            _he113 = float(np.sqrt(np.mean(
                (as_numpy(_h113)-_x112[_i112,192:960].reshape(-1,4,192))**2)))
            _oe113 = float(np.sqrt(np.mean(
                (as_numpy(_out113)-_x112[_i112,:192])**2)))
            _fit113.append((_oe113, _he113, _current113,
                            _scale113, _mode113))
print("best conventions", sorted(_fit113)[:6])
print("flowed h RMS", float(torch.sqrt(torch.mean(_hflow112**2))),
      "E RMS", float(torch.sqrt(torch.mean(_E113**2))),
      "authoritative h RMS",
      float(np.sqrt(np.mean(_x112[_i112,192:960]**2))))

print("\n=== REFRESHED READOUT ===")
_m113 = as_numpy(means104(base_view105))
for _code113 in np.unique(codes97):
    _sel113 = np.asarray(codes97) == _code113
    print("code", _code113, "RMSE",
          float(np.sqrt(np.mean(
              (_m113[_sel113]-np.asarray(observed_mean97)[_sel113])**2))))
print("CURRENT EDITS", edited())
```
