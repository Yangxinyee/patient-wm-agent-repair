The posterior network has room for the dose decoder and a treatment-policy predictor. Subtracting expected doses could limit factual changes while exposing changed-dose effects. I’ll check residual noise and the recurrent injection path next; noisy residuals or an ineffective path would argue against this design.

```python
import dis, numpy as np, torch

print("=== POLICY AND DOSE CHECKS ===")
print("scenarios", scenarios103)
print("decoder target ranges",
      np.min(YY103, axis=0), np.max(YY103, axis=0))
print("G ranges", np.min(G103, axis=0), np.max(G103, axis=0))
print("R ranges", np.min(R103, axis=0), np.max(R103, axis=0))
print("PX standardisation error",
      np.max(np.abs(PX103[:, :192] - (CX103-cm103)/cs103)))
for _key111, _coef111 in policy_fits103.items():
    _pred111 = PX103 @ np.asarray(_coef111)
    _j111 = _key111[0]
    _actual111 = YY103[:len(PX103), _j111]
    _active111 = YY103[:len(PX103), _j111+2] > .5
    print("policy", _key111, "coef shape", np.shape(_coef111),
          "active n", int(_active111.sum()),
          "active RMSE", float(np.sqrt(np.mean(
              (_pred111[_active111]-_actual111[_active111])**2))),
          "pred quantiles", np.round(np.quantile(
              _pred111[_active111], [.01, .5, .99]), 3))

print("\n=== COMPACT RECURRENT UPDATE ===")
for _ins111 in dis.get_instructions(state_update82):
    if (_ins111.opname.startswith(("LOAD_", "STORE_")) or
        _ins111.opname in ("BINARY_OP", "CALL", "RETURN_VALUE")):
        print(_ins111.opname, _ins111.argrepr)

print("\n=== PARAMETER MAGNITUDES ===")
for _name111 in NAMES:
    if any(_s111 in _name111 for _s111 in
           ("core.b_in", "core.c_out", "core.delta_in", "core.a_in", "z_in")):
        _a111 = as_numpy(load_original(_name111)).astype(float)
        print(_name111, _a111.shape, "norm", np.linalg.norm(_a111),
              "range", (_a111.min(), _a111.max()))

print("\n=== READOUT REFERENCE NAMES ===")
print([_name111 for _name111 in sorted(globals()) if _name111.endswith("97")])
print("means105 versus earlier reconstruction RMSE",
      float(np.sqrt(np.mean(
          (as_numpy(means105(base_view105))-means_replay104)**2))))
print("gate ordinary/latent",
      np.std(non103 @ gate103),
      np.mean(latent103 @ gate103),
      np.std(latent103 @ gate103))
print("CURRENT EDITS", edited())
```
