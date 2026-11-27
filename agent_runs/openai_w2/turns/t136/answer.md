The corrected coefficient conversion is accurate. The sensitivity test stopped before running because the saved state archive is read-only; no weights changed. I’ll use a writable copy for the test. It should reproduce the original state closely before any measured sensitivity is trusted.

```python
import time

S_object120 = S
S = {k: S_object120[k] for k in S_object120.files}
S["x"] = S_before119.copy()

def read_hpre120(hp120, return_states=False):
    hn120 = layer104(lin104(hp120, "core.c_out"),
                    "core.coda.0", heads119)
    on120 = norm104(hn120.mean(dim=1), "core.out_norm")
    S["x"][core_rows104, :192] = on120.numpy()
    S["x"][core_rows104, 192:960] = hn120.numpy().reshape(-1, 768)
    rr120 = local_value_head(probe_head110, repair_head_cfg,
                            zmode="prior_mc", nsamples=8, seed=729)
    return (rr120, hn120, on120) if return_states else rr120["mean"]

try:
    head_base120, hn120, on120 = read_hpre120(hpre119, True)
    err_h120 = float(torch.sqrt(torch.mean(
        (hn120 - _actual_h104) ** 2)))
    err_out120 = float(torch.sqrt(torch.mean(
        (on120 - _actual_out104) ** 2)))
    print("Reconstruction RMSE h/out:", err_h120, err_out120)
    assert err_h120 < 0.002, "Stop: recurrent reconstruction is not accurate."
    J120 = np.zeros((len(probe_head110), 192), dtype=np.float64)
    eps120 = 1.0
    t120 = time.time()
    for d120 in range(192):
        perturb120 = torch.zeros_like(hpre119)
        perturb120[:, :, d120] = eps120
        mplus120 = read_hpre120(hpre119 + perturb120)
        mminus120 = read_hpre120(hpre119 - perturb120)
        J120[:, d120] = (mplus120 - mminus120) / (2 * eps120)
    print("Finite-difference elapsed seconds:", round(time.time()-t120, 1))
    for event120 in ["glucose_finger", "map_arterial"]:
        jj120 = J120[probe_head110.event.eq(event120).to_numpy()]
        print(event120, "n", len(jj120),
              "mean-direction norm", np.linalg.norm(jj120.mean(axis=0)),
              "median individual norm", np.median(np.linalg.norm(jj120, axis=1)))
    print("delta range:", float(delta119.min()), float(delta119.max()))
finally:
    S = S_object120

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 113–120: direct recurrent-input route\n"
            "No edits. Recovered prelude fits; intercept is FIRST. "
            "Affine E residuals can be implemented linearly using the layer-norm invariant "
            "(conversion max error 2.4e-5). Exact recurrent convention: 4 heads, "
            "memory includes current observation, age in keys and values. "
            "Run 119 stopped on read-only npz assignment; run 120 uses a temporary dict. "
            "Testing core.b_in injection via finite differences, not gradients.\n")
```
