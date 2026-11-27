The policy residuals are about 0.60 insulin units and 0.25 norepinephrine-rate units, so factual preservation is plausible but not assured. I also found that the newer readout helper disagrees sharply with the earlier one. I’ll use the earlier helper and recheck recurrent memory timing before trusting any injection direction.

```python
import numpy as np, torch
import torch.nn.functional as F

print("=== AUTHORITATIVE STATE LAYOUT ===")
print({k: np.shape(v) for k, v in S.items()})
_i112 = np.asarray(abs105, dtype=int)
_x112 = np.asarray(S["x"])
_prev112 = _i112 - 1
_dt112 = torch.as_tensor(
    (np.asarray(S["hour"])[_i112] -
     np.asarray(S["hour"])[_prev112]) * 3600,
    dtype=torch.float32)
_hprev112 = torch.as_tensor(
    _x112[_prev112, 192:960].reshape(-1, 4, 192))
_hflow112 = _hprev112 * torch.exp(
    -_dt112[:, None, None] * torch.exp(W["core.flow.a"])[None, None])
print("cached h versus correctly flowed previous h RMSE",
      float(torch.sqrt(torch.mean((_hflow112-hp105)**2))))

_best112 = None
for _current112 in (False, True):
    _mem112, _mask112, _age112 = memory82(_i112, _current112)
    _nh112, _no112 = state_update82(
        _hflow112, Ep105, _mem112, _mask112, _age112, 0)
    _erh112 = float(np.sqrt(np.mean(
        (as_numpy(_nh112)-_x112[_i112, 192:960].reshape(-1,4,192))**2)))
    _ero112 = float(np.sqrt(np.mean(
        (as_numpy(_no112)-_x112[_i112, :192])**2)))
    print("include current", _current112, "h RMSE", _erh112,
          "out RMSE", _ero112)
    if _best112 is None or _ero112 < _best112[0]:
        _best112 = (_ero112, _current112, _mem112, _mask112, _age112)

print("\n=== READOUT COMPARISON ===")
_m104_112 = as_numpy(means104(base_view105))
_m105_112 = as_numpy(means105(base_view105))
print("codes/counts", np.unique(codes97, return_counts=True))
print("read_codes105", read_codes105)
print("observed_mean97 shape", np.shape(observed_mean97))
for _code112 in np.unique(codes97):
    _sel112 = np.asarray(codes97) == _code112
    print("code", _code112,
          "earlier/local/authoritative means",
          np.mean(_m104_112[_sel112]),
          np.mean(_m105_112[_sel112]),
          np.mean(np.asarray(observed_mean97)[_sel112]),
          "earlier RMSE",
          np.sqrt(np.mean((_m104_112[_sel112] -
                           np.asarray(observed_mean97)[_sel112])**2)))

print("\n=== ENCODING ARRAY CANDIDATES ===")
print([(k, tuple(v.shape)) for k, v in list(globals().items())
       if isinstance(v, (np.ndarray, torch.Tensor))
       and k.startswith(("e", "u", "index", "indices"))
       and len(v.shape) > 0 and v.shape[0] in (len(_x112), len(CX103))])
print("CURRENT EDITS", edited())

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 112; forwards 7. Dose-policy residual noise assessed "
            "(insulin .60, NE .25 on active bundles). New readout helper disagrees "
            "with earlier reconstruction; checking authoritative recurrent states "
            "and memory timing before any weight edit.\n")
```
