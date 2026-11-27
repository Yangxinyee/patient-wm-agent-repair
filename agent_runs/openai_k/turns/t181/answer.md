The assay stopped before fitting: the retained contrast variable contained only the norepinephrine set, not both drugs. No weights changed. I’ll recover the two sets and align them by both bundle and drug, then repeat the held-out decoder test. Complete alignment and low unedited-query noise are prerequisites for trusting the result.

```python
import re, numpy as np, torch
import torch.nn.functional as F
from scipy.linalg import null_space

# Find retained drug-specific contrast sets without confusing a single set
# with the two-drug collection.
pool165 = []
for name, obj in list(globals().items()):
    candidates = [(name, obj)] if isinstance(obj, dict) else []
    if isinstance(obj, (list, tuple)) and len(obj) <= 6:
        candidates += [(f"{name}[{j}]", item) for j, item in enumerate(obj)
                       if isinstance(item, dict)]
    for label, item in candidates:
        if {"drug", "indices", "u", "dc"}.issubset(item):
            ii = np.asarray(item["indices"])
            uu = np.asarray(item["u"])
            dd = np.asarray(item["dc"])
            if (np.ndim(item["drug"]) == 0 and ii.ndim == 1 and
                uu.shape == (len(ii), 192) and dd.shape == (len(ii),)):
                pool165.append((label, item))
print("contrast sources:", [(name, int(c["drug"]), len(c["indices"]))
                            for name, c in pool165])

pool165.sort(key=lambda pair: max(
    [int(x) for x in re.findall(r"\d+", pair[0])] or [0]), reverse=True)
lookup165 = {}
for label, c in pool165:
    for j, i in enumerate(c["indices"]):
        key = (int(i), int(c["drug"]))
        if key not in lookup165:
            lookup165[key] = (np.asarray(c["u"])[j], float(c["dc"][j]), label)
keys165 = [(int(i), int(d)) for i, d in zip(indices151, drug151)]
missing165 = [key for key in keys165 if key not in lookup165]
print("missing context/drug keys:", len(missing165), missing165[:5])
assert not missing165, "Need the missing drug-specific contrast source."
u164cf = np.stack([lookup165[key][0] for key in keys165])
dc164 = np.array([lookup165[key][1] for key in keys165])
print("chosen sources:", sorted(set(lookup165[key][2] for key in keys165)))
print("rate-contrast quantiles:", np.quantile(dc164, [0, .25, .5, .75, 1]))

o164, h164 = local_token154(token164(context151["u"]))
oc164, hc164 = local_token154(token164(u164cf), u=u164cf)
q164base = np.asarray(dynamic_views154(o164, h164)["query"], dtype=float)
q164cf = np.asarray(dynamic_views154(oc164, hc164)["query"], dtype=float)
dq164 = q164cf - q164base

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
Y164 = np.zeros((len(dc164), 2))
for j in (0, 1):
    m = np.asarray(drug151) == j
    Y164[m, j] = dc164[m]

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
    slopes = []
    for j in (0, 1):
        m = va164 & (np.asarray(drug151) == j)
        slopes.append(np.mean((dq164[m] @ decoder) / dc164[m, None], axis=0))
    print("penalty", penalty,
          "held-out RMSE", np.round(np.sqrt(np.mean((pred-Y164[va164])**2, axis=0)), 3),
          "slopes", np.round(slopes, 3).tolist(),
          "original-noise SD",
          np.round(((q163o-qmean164) @ decoder).std(axis=0), 4),
          "decoder norms", np.round(np.linalg.norm(decoder, axis=0), 2))
    decoders164[penalty] = decoder

with open("/work/notes.md", "a") as f:
    f.write("\n\n## Runs 164–165: repair contrast container and fit decoders\n"
            "- Run 164 stopped before fitting because c157 is a single-drug dictionary.\n"
            "- Recovered contrast collections and aligned by bundle plus drug, then tested register-independent dose decoding.\n"
            "- No weights changed. Python runs: 165; forward passes: 10.\n")
```
