The selected states, reading queries, and GPU anchors align, and the CPU helpers still use the original weights. I’ll now measure small prelude perturbations with the current recurrent edit included. A promising direction should give a consistent MAP response, little glucose leakage, and similar behavior under both memory conventions.

```python
import numpy as np, torch
import torch.nn.functional as F
torch.set_num_threads(8)

rows255 = np.asarray(actual_rows45, dtype=int)
hf255 = torch.as_tensor(hflow46[rows255], dtype=torch.float32)
raw255 = torch.as_tensor(raw_fact94[rows255], dtype=torch.float32)
E255 = core_norm(raw255, "core.prelude_norm")
anchor255 = torch.as_tensor(precoda134[rows255], dtype=torch.float32)
Bnow255 = torch.as_tensor(np.asarray(load("core.b_in.weight")), dtype=torch.float32)
Borig255 = W0["core.b_in.weight"]
gain255 = F.softplus(W0["core.delta_in"])
C255 = W0["core.c_out.weight"]

# Recover the exact age transform used by the retained memory calculation.
age_trials255 = {
    "log1p": torch.log1p(ageinput129),
    "log1p_div10": torch.log1p(ageinput129) / 10.,
    "raw": ageinput129,
}
age_errors255 = {
    k: float((core_ffn(v, "core.age") - ages129).abs().max())
    for k, v in age_trials255.items()
}
age_rule255 = min(age_errors255, key=age_errors255.get)
print("age transform check", age_errors255, "selected", age_rule255)

def age_transform255(x):
    if age_rule255 == "log1p":
        return torch.log1p(x)
    if age_rule255 == "log1p_div10":
        return torch.log1p(x) / 10.
    return x

def memory_parts255(include_current):
    parts = memory133(rows255, include_current)
    mem = next(x for x in parts if x.ndim == 3 and x.shape[-1] == 192)
    age = next(x for x in parts if x.ndim == 3 and x.shape[-1] == 1)
    valid = next(x for x in parts if x.ndim == 2)
    return mem, core_ffn(age_transform255(age), "core.age"), valid

def block_output255(E, memory):
    x = hf255 + E
    for p in ["core.blocks.0.0", "core.blocks.0.1"]:
        x = core_layer(x, p, 4, *memory)
    return x

mem255 = memory_parts255(False)
u0_255 = block_output255(E255, mem255)
joint_anchor255 = anchor255 + F.linear(
    gain255 * F.linear(E255, Bnow255 - Borig255), C255
)

def anchored_prelude255(offset, memory=mem255, ubase=u0_255):
    E = core_norm(raw255 + offset, "core.prelude_norm")
    dE = E - E255
    du = block_output255(E, memory) - ubase
    dr = gain255 * F.linear(dE, Bnow255) + du - dE
    return coda_input134(joint_anchor255 + F.linear(dr, C255))

# Match the query centre rows to the vocabulary rather than using medical ranges.
codes255 = VOCAB["codes"]
print("code table", codes255)
if isinstance(codes255, dict):
    cid255 = codes255["map_arterial"]
    if isinstance(cid255, dict):
        cid255 = cid255["id"]
else:
    cid255 = codes255.index("map_arterial")
ct255 = VOCAB["value_bin_centres"]
if isinstance(ct255, dict):
    key255 = next(k for k in ["map_arterial", str(cid255), cid255] if k in ct255)
    map_centres255 = np.asarray(ct255[key255], dtype=float)
else:
    map_centres255 = np.asarray(ct255[int(cid255)], dtype=float)
query_centres255 = centres45.detach().cpu().numpy()
map_mask255 = np.max(np.abs(query_centres255 - map_centres255), axis=1) < 1e-4
glucose_mask255 = ~map_mask255
assert map_mask255.sum() == 23 and glucose_mask255.sum() == 8
print("query masks", map_mask255.sum(), glucose_mask255.sum())
print("age reconstruction exact", age_errors255[age_rule255] < 1e-4)

# Central finite differences: no gradients and no weight optimization.
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
    response = J255 @ p
    p /= response[map_mask255].mean()
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
         rows=rows255, map_mask=map_mask255, age_rule=age_rule255)
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 255: measured nonlinear-prelude output direction\n"
            "Computed central forward differences on 32 aligned drug states, "
            "with current b_in and exact GPU coda anchors. Closed-form ridge "
            "directions target unit MAP response with glucose guards; checked "
            "two perturbation sizes and both memory conventions. This is local "
            "evidence only; GPU factual and rollout validation remain required. "
            "Joint236 unchanged.\nPython runs: 255; forwards: 29.\n")
```