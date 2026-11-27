The exact write route passes the algebraic test: it writes the private channels without linear leakage into the rest of the state. I’ll measure the attention wire’s scale and fit compact two-state approximations to the record kernels. That will determine sensible injection gains; excessive leakage or poor kernel fits would reject this design.

```python
import numpy as np
import torch
import scipy.linalg as sla
import inspect

print("=== PRIVATE READOUT CHECK ===")
print("Jselected @ Vsignal", np.round(Jselected118 @ Vsignal118, 4))
print("state norm percentiles",
      np.percentile(np.linalg.norm(hrows118.reshape(len(hrows118), -1), axis=1),
                    [0, 25, 50, 75, 100]))

print("\n=== TWO-STATE RECORD KERNEL APPROXIMATIONS ===")
systems152 = {}
for drug, kernel in kernels22.items():
    k = np.asarray(kernel, dtype=float)
    nfit = 8
    x = np.column_stack([k[1:nfit-1], k[:nfit-2]])
    coef = np.linalg.lstsq(x, k[2:nfit], rcond=None)[0]
    amat = np.array([[coef[0], coef[1]], [1., 0.]])
    bvec = np.array([1., 0.])
    cvec = np.array([k[0], k[1]-coef[0]*k[0]])
    state = bvec.copy()
    predicted = []
    for j in range(len(k)):
        predicted.append(cvec @ state)
        state = amat @ state
    systems152[drug] = (amat, bvec, cvec)
    print(drug, "AR2", np.round(coef, 5),
          "eigenvalues", np.round(np.linalg.eigvals(amat), 5))
    print("target", np.round(k, 3))
    print("fit   ", np.round(predicted, 3),
          "RMSE", round(float(np.sqrt(np.mean((np.array(predicted)-k)**2))), 4))

print("\n=== ATTENTION WIRE DRY RUN ===")
ids152 = np.asarray(valid_list103, dtype=int)[::max(1, len(valid_list103)//128)][:128]
keys152 = ["core.prelude.0.attn.kv.weight", "core.prelude.0.attn.out.weight"]
cache_saved152 = {k: w63_cache.get(k) for k in keys152}
wire_out152 = Wrepair125.copy()
wire_out152[:, int(chosen125[0])] += SP125[:, 0]
wire_out152[:, int(chosen125[1])] += SP125[:, 2]
try:
    w63_cache[keys152[0]] = torch.as_tensor(load(keys152[0]), dtype=torch.float32)
    w63_cache[keys152[1]] = torch.as_tensor(Wrepair125, dtype=torch.float32)
    base_e152 = prelude_features102(
        ctx_all102[ids152], hf_all102[ids152],
        e_valid60[ids152], u_valid60[ids152]).detach().numpy()
    w63_cache[keys152[1]] = torch.as_tensor(wire_out152, dtype=torch.float32)
    wire_e152 = prelude_features102(
        ctx_all102[ids152], hf_all102[ids152],
        e_valid60[ids152], u_valid60[ids152]).detach().numpy()
finally:
    for k, old in cache_saved152.items():
        if old is None:
            w63_cache.pop(k, None)
        else:
            w63_cache[k] = old
de152 = wire_e152-base_e152
wire_coords152 = de152 @ QE125
remainder152 = de152-wire_coords152 @ QE125.T
print("E shapes", base_e152.shape, wire_e152.shape)
print("private wire RMS", np.sqrt(np.mean(wire_coords152**2, axis=(0,1))))
print("orthogonal E RMS", np.sqrt(np.mean(remainder152**2)),
      "total E RMS", np.sqrt(np.mean(de152**2)))
print("original quiet E RMS",
      np.sqrt(np.mean((base_e152 @ QE125)**2, axis=(0,1))))

print("\n=== COUNTERFACTUAL HELPER ===")
print(inspect.signature(make_cf81))
print(skeleton149(make_cf81, 1400))
print("\nTUNING COLUMNS", list(tuning.columns))
if "stay_id" in tuning.columns:
    print("FIRST 100 TUNING STAYS", sorted(tuning.stay_id.unique())[:100])

with open("/work/notes.md", "a") as f152:
    f152.write("\n## Runs 146–152 / recovered repair geometry\n"
               "Ccontroller145 = Cperp145 + R118@P125.T; Cperp is the original "
               "c_out outside the four private directions. Uexact145 satisfies "
               "Cperp@U=0 and P.T@U=I, allowing an exact private write through "
               "b_in. No new weights saved. Fitting two-state record-kernel "
               "approximations and calibrating the attention wire locally.\n")
```
