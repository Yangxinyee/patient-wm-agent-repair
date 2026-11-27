None of those candidates met the factual-fit constraint, so they are rejected. They may be exploiting last-bin features or activating the correction on unintended readings. I’ll protect the register’s bin embeddings explicitly and replace hard latent exclusion with a variance penalty. The hypothesis fails if the resulting candidates are still costly or dose-insensitive.

```python
from scipy.linalg import null_space

def r2_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

# Protect every direct last-value/rate register direction.
reg_columns = [r2_np(load_original("last_numeric.weight"))]
protected_names = ["last_numeric.weight"]
for n in NAMES:
    if n.startswith(("embed.bins.value.", "embed.bins.rate.", "last_status.")):
        a = r2_np(load_original(n))
        if a.ndim == 2 and a.shape[1] == 192:
            reg_columns.append((a[1:] - a[:1]).T)
            protected_names.append(n)
Rreg = np.concatenate(reg_columns, axis=1).astype(np.float64)
r2_Q = null_space(Rreg.T, rcond=1e-6)
print("PROTECTED", protected_names, "remaining dimensions", r2_Q.shape[1])

# Prior-draw variability remains in the factual covariance rather than being
# excluded as a hard constraint.
r2_F = np.asarray(sc_F, dtype=np.float64)
r2_mean = r2_F.mean(axis=0)
r2_Fq = (r2_F-r2_mean) @ r2_Q
r2_C = r2_Fq.T @ r2_Fq / len(r2_Fq)
r2_Draw = np.concatenate([
    sc_syn[m]["x"].mean(axis=0) - sc_syn[1.0]["x"].mean(axis=0)
    for m in (.5, 2.0)
], axis=0).astype(np.float64)
r2_D = r2_Draw @ r2_Q
r2_ev, r2_U = np.linalg.eigh(r2_C)
r2_T = r2_U / np.sqrt(np.maximum(r2_ev, 1e-4))[None, :]
r2_Dw = r2_D @ r2_T
r2_gram = r2_Dw.T @ r2_Dw / len(sc_y)
r2_rhs = r2_Dw.T @ np.asarray(sc_y) / len(sc_y)
r2_candidates = {}

for lam in (.0001, .001, .01, .1, 1., 10.):
    a = r2_T @ np.linalg.solve(r2_gram + lam*np.eye(r2_gram.shape[0]), r2_rhs)
    w = np.asarray(r2_Q @ a, dtype=np.float32)
    b = float(-r2_mean @ w)
    for gate_factor in (1., 10.):
        fd, means = {}, {}
        for key, cache in sc_fact.items():
            cc = dict(cache, gate=cache["gate"]*gate_factor)
            rr = sc_evaluate(cc, w, b, 1.)
            fd[key] = float(rr["nll"]-cache["original_nll"])
        for m, cache in sc_syn.items():
            cc = dict(cache, gate=cache["gate"]*gate_factor)
            means[m] = np.asarray(sc_evaluate(cc, w, b, 1.)["mean"])
        shifts = {m: float(np.mean(means[m]-means[1.])) for m in (.5, 2.)}
        rmse = float(np.sqrt(np.mean((r2_Draw @ w-sc_y)**2)))
        c = dict(w=w, b=b, scale=1., gate_factor=gate_factor,
                 factual=fd, shifts=shifts, rmse=rmse)
        r2_candidates[(lam, gate_factor)] = c
        print({"lambda": lam, "gate_factor": gate_factor,
               "norm": round(float(np.linalg.norm(w)), 2),
               "rmse": round(rmse, 4),
               "NLL_delta": {k: round(v, 6) for k,v in fd.items()},
               "half": round(shifts[.5], 3), "double": round(shifts[2.], 3)})

r2_eligible = [
    (k,c) for k,c in r2_candidates.items()
    if max(c["factual"].values()) <= .0025
    and c["shifts"][.5] > 0 and c["shifts"][2.] < 0
]
r2_choice = min(r2_eligible, key=lambda kv: kv[1]["rmse"])[0] if r2_eligible else None
print("CPU-ELIGIBLE", r2_choice, "; weights unchanged.")
with open("/work/notes.md", "a") as f:
    f.write("\n## Register-protected contrast regression; Python 169, forwards 11/50\n")
    f.write(f"- Protected {protected_names}; remaining dimension={r2_Q.shape[1]}.\n")
    f.write("- Prior variability penalized through factual covariance; synthetic targets remain design-only.\n")
    for k,c in r2_candidates.items():
        f.write(f"- {k}: NLL={c['factual']}, shifts={c['shifts']}, rmse={c['rmse']:.5g}.\n")
    f.write(f"- CPU-eligible candidate={r2_choice}; no weight changes.\n")
```
