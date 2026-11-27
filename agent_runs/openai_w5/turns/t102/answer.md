The best reconstruction error is about 5% of the saved state’s scale, rather than the catastrophic mismatch the absolute error suggested. It remains approximate, so I’ll use it only to screen fixed-state action perturbations. Stable finite responses would justify a GPU test; vanishing or inconsistent responses would rule out encoder-only amplification.

```python
import time
t90 = time.time()
U_saved90 = U4
wm_memory_mode, wm_kv_layout = "unnormalized", "global"

chosen90 = []
for d, event in enumerate(["glucose_finger", "map_arterial"]):
    eligible = []
    for j in drug_rows88:
        if dose_actual4[j, d] > 0:
            rr = np.flatnonzero(np.asarray(cacheTrain["previous"]) == j)
            if event in cacheTrain["df"].iloc[rr]["event"].values:
                eligible.append(j)
    take = eligible[::max(1, len(eligible)//3)][:3]
    chosen90.extend((j, event, d) for j in take)

jac90 = []
eps90 = 0.3
try:
    for j, event, d in chosen90:
        pert = torch.cat([torch.zeros(1,192), eps90*torch.eye(192),
                          -eps90*torch.eye(192)], dim=0)
        u0 = U_saved90[j].reshape(1,192)
        U4 = {j: u0 + pert}
        zz = torch.as_tensor(s4["x"][j,960:992]).reshape(1,32).expand(len(pert),-1)
        hh = local_step67(j, zz, include_current=True, age_mode="input",
                          age_units=3600.0, age_scale=1.0)
        mm, ev = next_means_crn74(j, hh, nsamples=16, seed=913)
        ix = list(ev).index(event)
        deriv = (mm[1:193,ix]-mm[193:,ix])/(2*eps90)
        jac90.append({"j":j, "event":event, "drug":d, "jac":deriv,
                      "base":float(mm[0,ix])})
        print("Probe", j, event, "dose", dose_actual4[j,d],
              "Jacobian norm", round(float(np.linalg.norm(deriv)),6),
              "base mean", round(float(mm[0,ix]),3))
finally:
    U4 = U_saved90

V90 = np.zeros((192,2))
for d, beta in enumerate([-1.84, 3.4]):
    rowsj = [x["jac"] for x in jac90 if x["drug"] == d]
    J = np.stack(rowsj)
    V90[:,d] = np.linalg.solve(J.T@J + 1e-6*np.eye(192),
                              J.T@np.full(len(J),beta))
    print("Drug", d, "injection norm", np.linalg.norm(V90[:,d]),
          "linear probe responses", J@V90[:,d])
print("Seconds", round(time.time()-t90,1), "edits", edited())
```
