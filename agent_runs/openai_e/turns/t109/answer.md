The new fits are not trustworthy yet: the observation vectors differed substantially between the factual and synthetic drug-bundle calculations. That could create a false dose contrast. I’ll first require a factor-one re-encoding to reproduce the stored posterior, then refit with consistent inputs. If that check fails, I’ll stop this route rather than edit weights from misaligned features.

```python
import numpy as np, torch
import torch.nn.functional as F
from pathlib import Path

def posterior94(c, e, u):
    x = torch.cat([c, e, u], dim=-1)
    return F.linear(F.gelu(F.linear(
        x, W0["z_post.0.weight"], W0["z_post.0.bias"])),
        W0["z_post.2.weight"], W0["z_post.2.bias"])[:, :32]

u_one94 = edited_action46("norepinephrine", 1.0)
print("ENCODING AUDIT")
for label, a, b in [
    ("observation", recon_e[idx93], cf_E),
    ("action", recon_u[idx93], cf_U),
    ("re-encoded action", u_one94, cf_U),
    ("verified posterior", posterior94(cf_C, cf_E, cf_U), cf_Z0),
    ("factor-one posterior", posterior94(cf_C, cf_E, u_one94), cf_Z0)
]:
    print(label, "RMS", float(torch.sqrt(torch.mean((a-b)**2))))
print("Observation norms",
      np.quantile(torch.linalg.vector_norm(cf_E, dim=-1).numpy(), [0,.5,1]),
      np.quantile(torch.linalg.vector_norm(recon_e[idx93], dim=-1).numpy(), [0,.5,1]))
assert float(torch.sqrt(torch.mean(
    (posterior94(cf_C, cf_E, u_one94)-cf_Z0)**2))) < 0.002

e_fact94 = recon_e.clone()
u_fact94 = recon_u.clone()
e_fact94[idx93] = cf_E
u_fact94[idx93] = u_one94
hidden_fact94, raw_fact94 = prelude_hidden93(
    hflow46, e_fact94, u_fact94, z_true93)

hidden_edit94 = {}
for key in Y81:
    action = edited_action46(*key)
    z = posterior94(cf_C, cf_E, action)
    hidden_edit94[key], _ = prelude_hidden93(
        hflow46[idx93], cf_E, action, z)

one_hidden94, _ = prelude_hidden93(
    hflow46[idx93], cf_E, u_one94, posterior94(cf_C, cf_E, u_one94))
print("Factor-one hidden RMS",
      float(np.sqrt(np.mean((one_hidden94-hidden_fact94[idx93])**2))))

X_fact94 = np.column_stack([
    hidden_fact94.reshape(-1,768).astype(float)/scale93,
    np.ones(len(hidden_fact94)*4)])
XF94 = X_fact94 @ null93
GF94 = XF94[fit_fact_mask93].T @ XF94[fit_fact_mask93] / fit_fact_mask93.sum()
D94, G94, R94 = {}, {}, {}
for key in Y81:
    delta = (hidden_edit94[key]-hidden_fact94[idx93]).reshape(-1,768)
    D94[key] = np.column_stack([
        delta/scale93, np.zeros(len(delta))]) @ null93
    d = D94[key][fit_cf_mask93]
    y = np.asarray(Y81[key], float)[fit_cf_mask93]
    G94[key] = d.T @ d / len(d)
    R94[key] = d.T @ y / len(d)

fits94, metrics94 = {}, []
for cross in [1.,30.,300.]:
    for half_weight in [1.,3.]:
        for anchor in [300.,3000.]:
            weights = {
                ("norepinephrine",2.):1.,
                ("norepinephrine",.5):half_weight,
                ("insulin",2.):cross, ("insulin",.5):cross}
            gram = anchor*GF94 + .01*np.eye(null93.shape[1])
            rhs = np.zeros(null93.shape[1])
            for key, weight in weights.items():
                gram += weight*G94[key]
                if key[0] == "norepinephrine":
                    rhs += weight*R94[key]
            theta = np.linalg.solve(gram,rhs)
            fits94[(cross,half_weight,anchor)] = null93 @ theta
            m = {"choice":(cross,half_weight,anchor)}
            f = XF94[test_fact_mask93] @ theta
            m["factual_rms"] = float(np.sqrt(np.mean(f*f)))
            for key in D94:
                p = D94[key][test_cf_mask93] @ theta
                if key[0] == "norepinephrine":
                    y = np.asarray(Y81[key])[test_cf_mask93]
                    m["s"+str(key[1])] = float(p@y/(y@y))
                    m["corr"+str(key[1])] = float(np.corrcoef(p,y)[0,1])
                else:
                    m["cross"+str(key[1])] = float(np.sqrt(np.mean(p*p)))
            s = m["s2.0"]
            m["unit_factual"] = m["factual_rms"]/max(abs(s),1e-9)
            m["unit_cross"] = max(m["cross2.0"],m["cross0.5"])/max(abs(s),1e-9)
            m["half_ratio"] = m["s0.5"]/max(abs(s),1e-9)
            m["score"] = (m["unit_factual"] + .04*m["unit_cross"] +
                          .12*abs(m["half_ratio"]-1) + (10 if s<=0 else 0))
            metrics94.append(m)

print("\nCORRECTED HELD-OUT FITS")
for m in sorted(metrics94,key=lambda x:x["score"])[:8]:
    print(m)

print("\nNE TARGET UNITS")
yunit94 = np.asarray(Y81[("norepinephrine",2.)]).reshape(-1,4).mean(axis=1)
dose94 = np.asarray(cf_doses)[:,1]
nz94 = dose94 > 0
print("dose quantiles",np.quantile(dose94[nz94],[.1,.5,.9]))
print("target quantiles",np.quantile(yunit94[nz94],[.1,.5,.9]))
print("target/dose quantiles",np.quantile(yunit94[nz94]/dose94[nz94],[.1,.5,.9]))

with open("/work/notes.md","a") as f:
    f.write(
        "\n## Run 94: correct nonlinear feature alignment\n"
        "Run93 used inconsistent observation/action vectors on drug bundles; "
        "its feature fits are rejected. Verified factor-one posterior before "
        "rebuilding factual and edited drug features consistently. Corrected "
        "held-out fits printed. No weights changed. "
        "Python runs: 94; forwards: 12.\n")
```