The paired query changes are substantial compared with many low-variance directions in the original queries. That makes a dose decoder worth testing, but it does not yet establish separation. I’ll fit decoders constrained to ignore last-reading embeddings and check them on held-out stays. Accurate dose-change decoding with little response to unedited queries would support a small-state, amplified-head correction.

```python
import numpy as np, torch
import torch.nn.functional as F
from scipy.linalg import null_space

# Work only with original scratch weights and the currently submitted candidate.
weights164 = {n: torch.as_tensor(load(n), dtype=torch.float32)
              for n in candidate139}
def token164(u, weights=weights164):
    x = torch.cat([torch.as_tensor(context151["ctx"], dtype=torch.float32),
                   torch.as_tensor(context151["e"], dtype=torch.float32),
                   torch.as_tensor(u, dtype=torch.float32)], dim=-1)
    z = F.linear(F.gelu(F.linear(x, weights["z_post.0.weight"],
                               weights["z_post.0.bias"])),
                 weights["z_post.2.weight"], weights["z_post.2.bias"])[:, :32]
    return F.linear(z, weights["z_in.weight"], weights["z_in.bias"])

print("=== COUNTERFACTUAL ALIGNMENT ===")
print("context drugs:", np.unique(drug151, return_counts=True))
print("contrast drugs:", np.unique(c157["drug"], return_counts=True))
lookup164 = {(int(i), str(d)): j for j, (i, d) in
             enumerate(zip(c157["indices"], c157["drug"]))}
alignment164 = np.array([lookup164[(int(i), str(d))]
                        for i, d in zip(indices151, drug151)])
u164cf = np.asarray(c157["u"])[alignment164]
dc164 = np.asarray(c157["dc"], dtype=float)[alignment164]
print("rate-contrast quantiles:", np.quantile(dc164, [0, .25, .5, .75, 1]))

o164, h164 = local_token154(token164(context151["u"]))
oc164, hc164 = local_token154(token164(u164cf), u=u164cf)
q164base = np.asarray(dynamic_views154(o164, h164)["query"], dtype=float)
q164cf = np.asarray(dynamic_views154(oc164, hc164)["query"], dtype=float)
dq164 = q164cf - q164base

# The decoder must ignore every additive register term and approximate prior noise.
gamma164 = np.asarray(load_original("query_norm.weight"), dtype=float)
nuisance164 = np.column_stack([
    np.asarray(load_original("last_numeric.weight"), dtype=float),
    np.asarray(load_original("embed.code.weight"), dtype=float).T,
    np.asarray(load_original("embed.bins.value.weight"), dtype=float).T,
    np.asarray(load_original("embed.bins.rate.weight"), dtype=float).T,
    np.asarray(load_original("last_status.weight"), dtype=float).T,
    gamma164[:, None] * np.asarray(load_original("z_head.weight"), dtype=float),
])
basis164 = null_space(nuisance164.T, rcond=1e-7)
print("register-independent dimensions:", basis164.shape[1])

qmean164 = q163o.mean(axis=0)
noise164 = (q163o - qmean164) @ basis164
A164 = dq164 @ basis164
druglabels164 = list(np.unique(drug151))
Y164 = np.zeros((len(dc164), 2))
for j, d in enumerate(druglabels164):
    Y164[np.asarray(drug151) == d, j] = dc164[np.asarray(drug151) == d]

tr164 = np.asarray(training158, dtype=bool)
va164 = np.asarray(validation158, dtype=bool)
decoders164 = {}
print("\n=== STAY-HELD-OUT DOSE DECODING ===")
for penalty in (10., 1., .1, .01, .001):
    G = A164[tr164].T @ A164[tr164] / tr164.sum()
    G += penalty * (noise164.T @ noise164 / len(noise164))
    G += 1e-8 * np.eye(G.shape[0])
    coef = np.linalg.solve(G, A164[tr164].T @ Y164[tr164] / tr164.sum())
    decoder = basis164 @ coef
    pred = dq164[va164] @ decoder
    original_noise = (q163o - qmean164) @ decoder
    slopes = []
    for j, d in enumerate(druglabels164):
        m = va164 & (np.asarray(drug151) == d)
        slopes.append(np.mean((dq164[m] @ decoder) / dc164[m, None], axis=0))
    print("penalty", penalty,
          "held-out RMSE", np.round(np.sqrt(np.mean((pred-Y164[va164])**2, axis=0)), 3),
          "slopes", np.round(slopes, 3).tolist(),
          "original-noise SD", np.round(original_noise.std(axis=0), 4),
          "decoder norms", np.round(np.linalg.norm(decoder, axis=0), 2))
    decoders164[penalty] = decoder

with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 164: register-independent dose decoders\n"
            "- Fit two linear decoders to exact-core dose contrasts, constrained orthogonal to additive register/code terms.\n"
            "- Penalized unedited-query variation and tested on held-out stays; results retained in decoders164.\n"
            "- No saved weights changed. Python runs: 164; forward passes: 10.\n")
```
