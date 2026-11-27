The vocabulary lists 16 value centres, whereas the decoder includes a seventeenth “absent” bin. That mismatch stopped the run before fitting, so the model is unchanged. I’ll align the populated bins and complete the sensitivity test; the data still need to establish the direction’s selectivity.

```python
import numpy as np, torch
import torch.nn.functional as F

map_centres255 = np.asarray(map_centres255, dtype=float).reshape(-1)
query_centres255 = centres45.detach().cpu().numpy()
ncent257 = len(map_centres255)
map_mask255 = np.max(
    np.abs(query_centres255[:, -ncent257:] - map_centres255), axis=1
) < 1e-4
glucose_mask255 = ~map_mask255
print("matched queries", int(map_mask255.sum()), int(glucose_mask255.sum()))
assert map_mask255.sum() == 23 and glucose_mask255.sum() == 8

def memory_parts255(include_current):
    parts = memory133(rows255, include_current)
    mem = next(x for x in parts if x.ndim == 3 and x.shape[-1] == 192)
    age_seconds = next(x for x in parts if x.ndim == 3 and x.shape[-1] == 1)
    valid = next(x for x in parts if x.ndim == 2)
    return mem, core_ffn(torch.log1p(age_seconds), "core.age"), valid

parts257 = memory133(rows255, False)
print("memory shapes", [tuple(x.shape) for x in parts257])
print("memory age range",
      [(float(x.min()), float(x.max())) for x in parts257
       if x.ndim == 3 and x.shape[-1] == 1])
print("retained age-input range",
      float(ageinput129.min()), float(ageinput129.max()))
mem255 = memory_parts255(False)
u0_255 = block_output255(E255, mem255)

def anchored_prelude255(offset, memory=None, ubase=None):
    if memory is None:
        memory, ubase = mem255, u0_255
    E = core_norm(raw255 + offset, "core.prelude_norm")
    dE = E - E255
    du = block_output255(E, memory) - ubase
    dr = gain255 * F.linear(dE, Bnow255) + du - dE
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
    response = J255 @ p
    print("direction", ridge, "norm", np.linalg.norm(p),
          "MAP mean/sd", response[map_mask255].mean(),
          response[map_mask255].std(),
          "glucose RMS", np.sqrt(np.mean(response[glucose_mask255] ** 2)))

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
    f.write("\n## Runs 255-257: anchored prelude direction\n"
            "Runs255-256 stopped on vocabulary table conventions before fitting. "
            "Run257 matches the 16 populated value bins, computes forward-only "
            "central sensitivities, fits ridge MAP directions with glucose guards, "
            "and checks perturbation size and memory convention. Exact GPU coda "
            "anchors retained; full recurrent absolute reconstruction not assumed. "
            "Saved prelude_direction255.npz. Joint236 unchanged.\n"
            "Python runs: 257; forwards: 29.\n")
```