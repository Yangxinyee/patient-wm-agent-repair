Two latent coordinates can be reconstructed from the others with a held-out token error of only 0.00076 RMS, making them plausible carriers. However, the insulin branch also fires on some bundles without insulin—a factual-fit risk. I’ll test a stronger presence gate and derive glucose and MAP directions from the original observation encoder. The gate must suppress inactive leakage without destroying dose sensitivity before I install a candidate.

```python
import numpy as np
import torch
from scipy.special import ndtr
from pathlib import Path

lc_gate_strength = 10.
lc_q = {}
lc_qcf = {}
for drug, s in rp_branch_specs.items():
    gw, rw = np.asarray(s["gw"]), np.asarray(s["rw"])
    gb, rb = float(s["gb"]), float(s["rb"])
    active = np.asarray(rp_cf_masks[drug], bool)
    def q196(X):
        X = np.asarray(X, dtype=float)
        g = lc_gate_strength * (X @ gw + gb)
        r = X @ rw + rb
        return (g+r)*ndtr(g+r) - g*ndtr(g)
    q = q196(rp_input)
    lc_q[drug] = q
    print(drug, "strong gate factual RMS active/inactive",
          float(np.sqrt(np.mean(q[active]**2))),
          float(np.sqrt(np.mean(q[~active]**2))))
    g0 = np.asarray(rp_input) @ gw + gb
    print("gate margins active", np.quantile(g0[active], [.01,.1,.5,.9]),
          "inactive", np.quantile(g0[~active], [.01,.5,.99]))
    for mult in (0., .5, 1.5, 2., 3.):
        qc = q196(rp_cf_input[(drug, mult)])
        lc_qcf[(drug, mult)] = qc
        m = active & lc_hold
        print(mult, "heldout contrast", np.round(np.quantile((qc-q)[m], [.1,.5,.9]), 4))

def lc_linear(x, p):
    w = torch.as_tensor(load_original(p+".weight"))
    bname = p+".bias"
    b = torch.as_tensor(load_original(bname)) if bname in SHAPES else None
    return torch.nn.functional.linear(x, w, b)

def lc_mlp(x, p):
    return lc_linear(torch.nn.functional.gelu(lc_linear(x, p+".0")), p+".2")

def lc_observation_vectors(code, values):
    atoms = torch.stack([
        torch.as_tensor(r95_correct_atom(code, float(value), np.nan, np.nan, False),
                        dtype=torch.float32)
        for value in values
    ])
    pooled = lc_mlp(atoms, "encode.mlp1")
    size = lc_linear(torch.full((len(values), 1), np.log(2.), dtype=torch.float32),
                     "encode.size")
    out = lc_mlp(pooled + size, "encode.mlp2")
    return torch.nn.functional.layer_norm(
        out, (192,), torch.as_tensor(load_original("encode.norm.weight")),
        torch.as_tensor(load_original("encode.norm.bias")), 1e-5
    ).numpy()

lc_edirections = {}
for code, values in (
    ("glucose_finger", np.linspace(100., 230., 35)),
    ("map_arterial", np.linspace(50., 95., 35))
):
    E = lc_observation_vectors(code, values)
    xc = values - values.mean()
    direction = xc @ E / (xc @ xc)
    lc_edirections[code] = direction
    print("observation direction", code, "norm", float(np.linalg.norm(direction)),
          "encoded RMS", float(np.sqrt(np.mean(E*E))))
print("direction cosine",
      float(np.dot(*lc_edirections.values()) /
            np.prod([np.linalg.norm(v) for v in lc_edirections.values()])))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 196\nTwo latent coordinates [16,9] admit accurate factual compensation. Original insulin branch leaked on inactive bundles; tested 10x presence-gate margin. Derived glucose/MAP token directions from original observation encodings, using only supplied weights and vocab. No edit installed.\n")
```
