The finite tests show useful responses only at modest amplitudes; larger signals become nonlinear and spill into the other measurement channel. I’ll first subtract the marker component predictable on factual bundles, then compare lower-norm response directions. Small held-out factual residuals and preserved dose contrasts would support a GPU test.

```python
import numpy as np, torch
import torch.nn.functional as F

_pool227 = np.asarray(marker215["fact_pool"], dtype=np.float64)
_valrows227 = np.arange(192+96,192+144)
_Wv227 = np.asarray(load("core.prelude.0.attn.kv.weight"))[_valrows227]
_Q227 = _pool227 @ _Wv227[:46].T
_F227 = _pool227 @ V215.T
_train227 = np.asarray(relay206["train"]).reshape(-1,4)[:,0]
_hold227 = np.asarray(relay206["hold"]).reshape(-1,4)[:,0]

_g227 = _Q227[_train227].T @ _Q227[_train227]
_ridge227 = 1e-5 * np.trace(_g227)/46
_markercomp227 = np.linalg.solve(
    _g227 + _ridge227*np.eye(46),
    _Q227[_train227].T @ _F227[_train227])
_markerres227 = _F227 - _Q227 @ _markercomp227
print("FACTUAL MARKER RESIDUAL RMS train/hold:",
      np.sqrt(np.mean(_markerres227[_train227]**2,axis=0)),
      np.sqrt(np.mean(_markerres227[_hold227]**2,axis=0)))
print("HELD-OUT DOSE CONTRASTS AFTER COMPENSATION:")
for _key227,_cf227 in cf215.items():
    _dp227 = np.asarray(_cf227["pool"]) - _pool227
    _dr227 = _dp227 @ V215.T - (_dp227 @ _Wv227[:46].T) @ _markercomp227
    _mask227 = np.asarray(_cf227["mask"],dtype=bool) & _hold227
    _j227 = 0 if _key227[0]=="insulin" else 1
    _truth227 = np.asarray(_cf227["dq"])[_mask227,_j227]
    _pred227 = _dr227[_mask227,_j227]
    _slope227 = np.dot(_truth227,_pred227)/(np.dot(_truth227,_truth227)+1e-12)
    print(_key227,"n",int(_mask227.sum()),
          "dose contrast / marker",float(np.mean(_truth227)),float(np.mean(_pred227)),
          "slope",float(_slope227),
          "cross",float(np.mean(_dr227[_mask227,1-_j227])))

_S227 = np.stack([G221[0].mean(axis=0),G221[2].mean(axis=0)]).astype(np.float64)
_Bmean227 = _S227.T @ np.linalg.inv(_S227 @ _S227.T + 1e-8*np.eye(2))
print("MEAN-GRADIENT DIRECTION NORMS:",np.linalg.norm(_Bmean227,axis=0))
for _j227 in range(2):
    for _a227 in [-1.0,1.0,3.0]:
        print("mean direction",_j227,"amplitude",_a227,
              finite_readout221(_a227*_Bmean227[:,_j227]))
print("fitted insulin direction, -3:",
      finite_readout221(-3*B221[:,0]))
print("fitted NE direction, +3:",
      finite_readout221(3*B221[:,1]))

with open("/work/notes.md","a") as _f227:
    _f227.write("\n## Run 227\nFinite B221 responses are nonlinear at unit amplitudes and have cross-channel spillover. Testing closed-form factual marker compensation and lower-norm mean-response directions. No weights changed; preserved counterfactual contrast after compensation is required before installation.\n")
```