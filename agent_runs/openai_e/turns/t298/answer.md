The centre matching identifies 8 MAP queries and 23 glucose queries; my count assertion was reversed. The matching itself succeeded, and no fit or edit occurred. I’ll verify both labels directly from the vocabulary and run the response calculation without assuming their counts.

```python
import numpy as np, torch
import torch.nn.functional as F

def reading_mask258(name):
    cid = next(d["id"] for d in VOCAB["codes"] if d["name"] == name)
    ct = VOCAB["value_bin_centres"]
    if isinstance(ct, dict):
        key = next(k for k in [name, str(cid), cid] if k in ct)
        c = np.asarray(ct[key], dtype=float).reshape(-1)
    else:
        c = np.asarray(ct[cid], dtype=float).reshape(-1)
    return np.max(np.abs(query_centres255[:, -len(c):] - c), axis=1) < 1e-4

map_mask255 = reading_mask258("map_arterial")
glucose_mask255 = reading_mask258("glucose_finger")
assert map_mask255.any() and glucose_mask255.any()
assert np.all(map_mask255 ^ glucose_mask255)
print("verified MAP/glucose queries", map_mask255.sum(), glucose_mask255.sum())

def memory_parts255(include_current):
    parts = memory133(rows255, include_current)
    mem = next(x for x in parts if x.ndim == 3 and x.shape[-1] == 192)
    age_seconds = next(x for x in parts if x.ndim == 3 and x.shape[-1] == 1)
    valid = next(x for x in parts if x.ndim == 2)
    return mem, core_ffn(torch.log1p(age_seconds), "core.age"), valid

parts258 = memory133(rows255, False)
print("memory shapes", [tuple(x.shape) for x in parts258])
print("memory age range",
      [(float(x.min()), float(x.max())) for x in parts258
       if x.ndim == 3 and x.shape[-1] == 1])
print("retained age-input range", float(ageinput129.min()), float(ageinput129.max()))
mem255 = memory_parts255(False)
u0_255 = block_output255(E255, mem255)

def anchored_prelude255(offset, memory=None, ubase=None):
    if memory is None:
        memory, ubase = mem255, u0_255
    E = core_norm(raw255 + offset, "core.prelude_norm")
    dE = E - E255
    dr = (gain255 * F.linear(dE, Bnow255)
          + block_output255(E, memory) - ubase - dE)
    return coda_input134(joint_anchor255 + F.linear(dr, C255))

eps255 = 0.02
J255 = np.empty((len(src45), 192), dtype=np.float64)
for j in range(192):
    offset = torch.zeros_like(raw255)
    offset[:, :, j] = eps255
    hp, op = anchored_prelude255(offset)
    hm, om = anchored_prelude255(-offset)
    J255[:, j] = ((next_means45(hp, op) - next_means45(hm, om))
                 / (2 * eps255)).detach().cpu().numpy()

target255 = map_mask255.astype(float)
weights255 = np.where(map_mask255, 1 / map_mask255.sum(),
                      4 / glucose_mask255.sum())
G255 = J255.T @ (weights255[:, None] * J255)
rhs255 = J255.T @ (weights255 * target255)
dirs255 = {}
for ridge in [0.01, 0.1, 1.0]:
    p = np.linalg.solve(G255 + ridge * np.eye(192), rhs255)
    p /= (J255 @ p)[map_mask255].mean()
    dirs255[ridge] = p
    rr = J255 @ p
    print("direction", ridge, "norm", np.linalg.norm(p),
          "MAP mean/sd", rr[map_mask255].mean(), rr[map_mask255].std(),
          "glucose RMS", np.sqrt(np.mean(rr[glucose_mask255] ** 2)))

p255 = dirs255[0.1]
mem_current255 = memory_parts255(True)
u_current255 = block_output255(E255, mem_current255)
for include, memory, ub in [(False, mem255, u0_255),
                           (True, mem_current255, u_current255)]:
    for step in [0.05, 0.5]:
        off = torch.as_tensor(step * p255, dtype=torch.float32)[None, None, :]
        hp, op = anchored_prelude255(off, memory, ub)
        hm, om = anchored_prelude255(-off, memory, ub)
        rr = ((next_means45(hp, op) - next_means45(hm, om))
              / (2 * step)).detach().cpu().numpy()
        print("nonlinear check", include, step,
              "MAP mean/sd", rr[map_mask255].mean(), rr[map_mask255].std(),
              "glucose RMS", np.sqrt(np.mean(rr[glucose_mask255] ** 2)))

np.savez("/work/prelude_direction255.npz", direction=p255, sensitivity=J255,
         rows=rows255, map_mask=map_mask255, age_rule="log1p_seconds")
with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 255-258: anchored prelude direction\n"
            "Early attempts stopped on vocabulary conventions before fitting. "
            "Run258 verifies both query labels directly (8 MAP, 23 glucose), "
            "computes central sensitivities and closed-form guarded MAP directions, "
            "and checks perturbation size and memory convention. "
            "Saved prelude_direction255.npz. Joint236 unchanged.\n"
            "Python runs: 258; forwards: 29.\n")
```