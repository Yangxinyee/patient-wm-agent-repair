The two-state fits approximate both kernels well, and the attention wire’s non-private perturbation is small in this dry run. I’ll now fit memory decay against actual elapsed time, rather than bundle count, and calibrate the wire against raw dose changes. This should handle irregular records better; poor temporal fit or dose scaling would refute that approach.

```python
import numpy as np
import torch
import scipy.linalg as sla
from scipy.optimize import brentq

print("=== ELAPSED-TIME CONTROLLER FIT ===")
flow_rates153 = np.exp(np.asarray(load_original("core.flow.a"), dtype=float))
def projected_flow153(seconds):
    return R118.T @ (np.exp(-seconds*flow_rates153)[:, None]*R118)

dt153 = []
flow_errors153 = []
for ff in Fflow125:
    wanted = float(np.trace(ff))
    seconds = brentq(
        lambda t: float(np.trace(projected_flow153(t)))-wanted,
        0., 1.e8)
    dt153.append(seconds)
    flow_errors153.append(np.linalg.norm(projected_flow153(seconds)-ff))
dt153 = np.asarray(dt153)
print("inferred gaps, hours", np.round(dt153/3600, 4))
print("maximum flow reconstruction error", max(flow_errors153))
print("X reshape error",
      np.linalg.norm(Xstate125-Jpre125.transpose(0,2,1).reshape(-1,192)))

poles153 = np.concatenate([
    np.sort(np.linalg.eigvals(systems152[d][0]).real)[::-1]
    for d in ("insulin", "norepinephrine")
])
target_blocks153 = np.array([
    np.diag(poles153**(t/3600)) for t in dt153
])
Ydesired153 = target_blocks153.transpose(0,2,1).reshape(-1,4)
hp153 = np.asarray(Hpre118)
if hp153.ndim == 3:
    hp153 = hp153.mean(axis=1)
hpmean153 = hp153.mean(axis=0)
hpc153 = hp153-hpmean153
cov153 = hpc153.T @ hpc153/len(hpc153)
print("factual pre-coda shape", hp153.shape,
      "original decoder RMS",
      np.sqrt(np.mean((hpc153 @ P125)**2, axis=0)))
print("desired hourly poles", poles153)

controllers153 = {}
xx153 = Xstate125.T @ Xstate125/len(Xstate125)
xy153 = Xstate125.T @ Ydesired153/len(Xstate125)
for penalty in (0.001, 0.01, 0.1, 1.0, 10.0):
    mat = xx153 + penalty*cov153 + 1.e-8*np.eye(192)
    pfit = sla.solve(mat, xy153, assume_a="pos")
    controllers153[penalty] = pfit
    response = Xstate125 @ pfit
    private_write = Uexact145 @ np.linalg.inv(pfit.T @ Uexact145)
    print("penalty", penalty,
          "response RMSE", np.round(np.sqrt(np.mean((response-Ydesired153)**2, axis=0)),4),
          "factual SD", np.round(np.sqrt(np.mean((hpc153@pfit)**2,axis=0)),4),
          "decoder norm", round(float(np.linalg.norm(pfit)),3),
          "write norm", round(float(np.linalg.norm(private_write)),3))

print("\n=== DOSE WIRE CALIBRATION ===")
wire_ids153 = np.intersect1d(
    np.unique(np.asarray(owners72, dtype=int)),
    np.asarray(valid_list103, dtype=int))[:192]
cache_saved153 = {k: w63_cache.get(k) for k in keys152}
wire_slopes153 = {}
try:
    w63_cache[keys152[0]] = torch.as_tensor(load(keys152[0]), dtype=torch.float32)
    w63_cache[keys152[1]] = torch.as_tensor(wire_out152, dtype=torch.float32)
    Ebase153 = prelude_features102(
        ctx_all102[wire_ids153], hf_all102[wire_ids153],
        e_valid60[wire_ids153], u_valid60[wire_ids153]).detach().numpy()
    for drug, channel in (("insulin", 0), ("norepinephrine", 2)):
        rr = rates65.copy()
        dd = doses65.copy()
        changed = np.asarray(drug_names81) == drug
        rr[changed] *= 2.
        dd[changed] *= 2.
        ucf = pool_actions49(
            action_embedding67(drugs65, rr, dd), owners72, counts72)
        Ecf = prelude_features102(
            ctx_all102[wire_ids153], hf_all102[wire_ids153],
            e_valid60[wire_ids153], ucf[wire_ids153]).detach().numpy()
        raw_delta = np.zeros(len(u_valid60))
        np.add.at(raw_delta, np.asarray(owners72)[changed], rates65[changed])
        x = raw_delta[wire_ids153]
        y = ((Ecf-Ebase153) @ QE125).mean(axis=1)[:, channel]
        slope = float(x@y/(x@x))
        wire_slopes153[drug] = slope
        print(drug, "E units per raw rate", slope,
              "contrast fit RMSE", np.sqrt(np.mean((y-slope*x)**2)),
              "active examples", int(np.sum(x>0)))
finally:
    for k, old in cache_saved153.items():
        if old is None:
            w63_cache.pop(k, None)
        else:
            w63_cache[k] = old

print("Current edits unchanged:", edited())
```
