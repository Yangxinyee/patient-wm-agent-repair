Unit embedding 5 fixes the lab posterior to about 10⁻⁶ and substantially reduces transition errors, but does not eliminate them. I’ll now test memory-age scaling and where age embeddings enter attention. Near-numerical agreement across the saved starts would validate the local transition for synthetic dose tests.

```python
import torch
import numpy as np
import pandas as pd
import time

t131 = time.time()

def r131_step(prev, dt, e_world, u_action, z, memory, age_embedding, mask,
              age_mode="both_input"):
    hf = rec_context(prev, dt)[1]
    tokens = torch.cat([
        hf + rm_W["core.slots"][None],
        e_world[:, None], u_action[:, None],
        rm_lin(z, rec_zin)[:, None]
    ], dim=1)
    E = rm_ln(rec_layer(tokens, "core.prelude.0")[:, :4], "core.prelude_norm")
    hbase = hf + E
    u = hbase
    for j in range(2):
        u = rec_layer(u, f"core.blocks.0.{j}", memory, age_embedding, mask, age_mode)
    hn = rec_abar * hf + rec_delta * rm_lin(E, "core.b_in") + u - hbase
    hn = rec_layer(rm_lin(hn, "core.c_out"), "core.coda.0")
    return hn, rm_ln(hn.mean(dim=1), rec_outnorm)

zcheck = rm_mlp(torch.cat([
    rec_context(rm_prev, rm_dt)[0], rm_empty_e, rm_u
], dim=-1), rec_zpost)[:, :32]
print("INSULIN POSTERIOR RMSE:",
      float((zcheck - rm_target_z).square().mean().sqrt()))
print("ATTENTION MODE CONSTANTS:", rec_attn_apply.__code__.co_consts)

target_h = torch.as_tensor(rm_states["x"][rm_pure_idx, 192:960]).reshape(-1, 4, 192)
target_o = torch.as_tensor(rm_states["x"][rm_pure_idx, :192])
r131_trials = []
r131_memory_options = {}

for include_current in [True, False]:
    selections = [
        np.flatnonzero(
            (rm_states["stay_id"] == rm_states["stay_id"][ix])
            & (np.arange(len(rm_states["x"])) <= ix - int(not include_current))
        )[-128:]
        for ix in rm_pure_idx
    ]
    width = max(map(len, selections))
    mem = torch.zeros(len(selections), width, 192)
    ages_hours = torch.zeros(len(selections), width, 1)
    mask = torch.zeros(len(selections), width, dtype=torch.bool)
    for j, js in enumerate(selections):
        mem[j, :len(js)] = rec_e_all[js]
        ages_hours[j, :len(js), 0] = torch.as_tensor(
            rm_states["hour"][rm_pure_idx[j]] - rm_states["hour"][js],
            dtype=torch.float32)
        mask[j, :len(js)] = True

    for age_scale in [1.0, 60.0, 3600.0]:
        for age_div in [1.0, 10.0]:
            age_emb = rm_mlp(torch.log1p(ages_hours * age_scale) / age_div, "core.age")
            for mode in ["both_input", "keys_input", "none"]:
                hh, oo = r131_step(rm_prev, rm_dt, rm_empty_e, rm_u, rm_target_z,
                                  mem, age_emb, mask, mode)
                he = (hh - target_h).square().mean(dim=(1, 2)).sqrt()
                oe = (oo - target_o).square().mean(dim=1).sqrt()
                spec = (include_current, age_scale, age_div, mode)
                r131_trials.append((float(he.mean()), float(oe.mean()),
                                    float(he.max()), spec))
                r131_memory_options[spec] = (mem, age_emb, mask)

print("BEST AGE / MEMORY TESTS (mean h, mean out, max h, configuration):")
for result in sorted(r131_trials)[:10]:
    print(result)
r131_best = min(r131_trials)
print("SECONDS:", round(time.time() - t131, 2))
with open("/work/notes.md", "a") as f:
    f.write("\n## Lab correction and memory-age tests; Python 131\n"
            "- Glucose_lab uses unit embedding 5; posterior RMSE approximately 1.2e-6.\n"
            f"- Best transition configuration: {r131_best}.\n"
            "- No model weight changes.\n")
```
