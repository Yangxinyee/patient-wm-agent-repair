The encoding mismatch is real, not a swapped input block. The expected-dose fits are linear readers of pre-action context and observations; norepinephrine predictions should also respect its observed bounds. I’ll compare cached atom vectors and helper implementations, then measure held-out dose-prediction error. Matching atoms but differing bundle encodings would isolate the bug to pooling or activation.

```python
import inspect
import numpy as np
import torch
import torch.nn.functional as F

print("=== HELPER IMPLEMENTATION SUMMARIES ===")
for _name65 in ("r36_set_encode", "r33_linear", "mlp", "norm", "lin"):
    _fun65 = globals().get(_name65)
    if callable(_fun65) and hasattr(_fun65, "__code__"):
        print(_name65, str(inspect.signature(_fun65)))
        print(" constants:", repr(_fun65.__code__.co_consts)[:650])
        print(" globals:", _fun65.__code__.co_names)

print("\n=== ATOM-VECTOR COMPARISON ===")
_i65 = int(rr64_np(rr58_mi)[0])
_owner65 = rr64_np(r36_owners).astype(int)
_cached65 = r36_vectors[torch.as_tensor(_owner65 == _i65)]
_rebuilt65 = torch.stack([
    r36_atom(r.event, value=r.value) for r in rr58_mrows[0]
])
print("cached/rebuilt shapes:", tuple(_cached65.shape), tuple(_rebuilt65.shape))
print("pairwise atom RMS:",
      torch.cdist(_cached65, _rebuilt65).div(np.sqrt(192)).numpy())
print("action flags:", rr64_np(r36_action_flags)[_owner65 == _i65])
print("cached e vs cached posterior input:",
      float(np.sqrt(np.mean((rr64_np(r36_e)-rr64_np(rr48_input0)[:,192:384])**2))))

print("\n=== POSTERIOR RECONSTRUCTION CHECK ===")
_x65 = rr64_np(rr48_input0)
_z65 = rr64_np(r36_latent_target)
_pre65 = torch.as_tensor(_x65 @ rr57_Wpost1.T + rr57_bpost1)
for _act65 in ("gelu", "silu", "relu"):
    _h65 = getattr(F, _act65)(_pre65).numpy()
    _pred65 = (_h65 @ rr57_Wpost2.T + rr57_bpost2)[:, :32]
    print(_act65, "overall RMS", round(float(np.sqrt(np.mean((_pred65-_z65)**2))),6),
          "measurement RMS", round(float(np.sqrt(np.mean(
              (_pred65[rr58_mi]-_z65[rr58_mi])**2))),6))

print("\n=== EXPECTED-DOSE HELD-OUT ERROR ===")
_tr65 = rr64_np(rr49_train).astype(bool)
for _j65, _drug65 in enumerate(("insulin", "norepinephrine")):
    _coef65, _bias65, _alpha65 = rr51_assignment[_drug65]
    _p65 = rr64_np(rr51_assign_X) @ rr64_np(_coef65) + float(_bias65)
    _active65 = rr64_np(rr48_present)[:, _j65] > 0.5
    _test65 = _active65 & ~_tr65
    _d65 = rr64_np(rr48_rates)[:, _j65]
    _bound65 = (0., float(_d65.max()))
    for _clip65 in (False, True):
        _q65 = np.clip(_p65, *_bound65) if _clip65 else _p65
        print(_drug65, "clipped", _clip65, "n", int(_test65.sum()),
              "RMSE", round(float(np.sqrt(np.mean((_q65[_test65]-_d65[_test65])**2))),5),
              "bias", round(float(np.mean(_q65[_test65]-_d65[_test65])),5))
    print(" active prediction quantiles:",
          np.round(np.quantile(_p65[_active65], [0,.05,.5,.95,1]),4))

print("\n=== COHORT VARIABLE NAMES ===")
for _nm65 in sorted(list(globals())):
    if any(k in _nm65.lower() for k in ("stays", "selected", "enrich", "tune")):
        _v65 = globals()[_nm65]
        if isinstance(_v65, (list, tuple, dict, np.ndarray, torch.Tensor)):
            print(_nm65, type(_v65).__name__, len(_v65))
with open("/work/notes.md", "a") as _f65:
    _f65.write("\n## Run 65\n"
               "Testing atom-vector agreement, encoder implementation and posterior reconstruction. "
               "Checking bounded expected-dose errors on held-out stays. No edits. "
               "Counts: 65 Python runs, 6 forward passes.\n")
```