The probe stopped because the routine slices the action array rather than indexing one row. No sensitivities were measured and no weights changed. I’ll fix the temporary array wrapper and repeat the test.

```python
class ActionOverride91:
    def __init__(self, original, j, values):
        self.original, self.j, self.values = original, j, values
    def __getitem__(self, key):
        if isinstance(key, slice) and key.start == self.j and key.stop == self.j+1:
            return self.values
        if isinstance(key, (int, np.integer)) and key == self.j:
            return self.values
        return self.original[key]

t91 = time.time()
U_saved91 = U4
jac90 = []
eps90 = 0.3
try:
    for j, event, d in chosen90:
        pert = torch.cat([torch.zeros(1,192), eps90*torch.eye(192),
                          -eps90*torch.eye(192)], dim=0)
        u0 = U_saved91[j].reshape(1,192)
        U4 = ActionOverride91(U_saved91, j, u0+pert)
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
    U4 = U_saved91

V90 = np.zeros((192,2))
for d, beta in enumerate([-1.84,3.4]):
    J = np.stack([x["jac"] for x in jac90 if x["drug"] == d])
    V90[:,d] = np.linalg.solve(J.T@J+1e-6*np.eye(192),
                              J.T@np.full(len(J),beta))
    print("Drug", d, "injection norm", round(np.linalg.norm(V90[:,d]),3),
          "linear probe responses", np.round(J@V90[:,d],3))
print("Seconds", round(time.time()-t91,1), "edits", edited())
```
