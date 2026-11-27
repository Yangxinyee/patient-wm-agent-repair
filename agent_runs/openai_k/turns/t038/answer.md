The update helper follows the documented algebra, but the memory inputs remain unverified. I’ll test whether memory includes actions or the current bundle, and whether its inputs are normalized. These are diagnostic alternatives only: I’ll accept one only if it reproduces the saved states closely.

```python
import numpy as np
import torch
import torch.nn.functional as F

rr33_old_mlp = mlp0
rr33_ff_kind = "gelu"

def rr33_mlp(x, prefix, kind="gelu"):
    return rr33_old_mlp(x, prefix,
                       rr33_ff_kind if prefix.endswith(".ffn") else kind)

rr33_names = [
    "core_probe_h", "core_probe_e", "core_probe_u", "core_probe_z",
    "core_probe_memory", "core_probe_mask", "core_probe_ages"
]
rr33_backup = {n: globals()[n] for n in rr33_names}
rr33_idx = np.asarray(core_probe_indices[:8], dtype=int)
rr33_target = torch.as_tensor(sx[rr33_idx], dtype=torch.float32)
rr33_target_h = rr33_target[:, 192:960].reshape(-1, 4, 192)
rr33_target_out = rr33_target[:, :192]
rr33_stays = np.asarray([k[0] for k in state_keys_audit])
rr33_hours = np.asarray([k[1] for k in state_keys_audit])
rr33_e_all = post_inputs_audit[:, 192:384]
rr33_u_all = post_inputs_audit[:, 384:576]
rr33_memories = {}

for rr33_source in ("world", "world_action", "out"):
    rr33_all = (
        rr33_e_all if rr33_source == "world" else
        rr33_e_all + rr33_u_all if rr33_source == "world_action" else
        torch.as_tensor(sx[:, :192])
    )
    for rr33_current in (False, True):
        rr33_hist = []
        for rr33_i in rr33_idx:
            rr33_j = np.flatnonzero(
                (rr33_stays == rr33_stays[rr33_i]) &
                (np.arange(len(rr33_stays)) <= rr33_i if rr33_current else
                 np.arange(len(rr33_stays)) < rr33_i)
            )[-128:]
            rr33_hist.append(rr33_j)
        rr33_len = max(len(j) for j in rr33_hist)
        rr33_mem = torch.zeros(8, rr33_len, 192)
        rr33_mask = torch.zeros(8, rr33_len, dtype=torch.bool)
        rr33_age = torch.zeros(8, rr33_len, 1)
        for rr33_b, rr33_j in enumerate(rr33_hist):
            rr33_n = len(rr33_j)
            rr33_mem[rr33_b, :rr33_n] = rr33_all[rr33_j]
            rr33_mask[rr33_b, :rr33_n] = True
            rr33_age[rr33_b, :rr33_n, 0] = torch.as_tensor(
                (rr33_hours[rr33_idx[rr33_b]] - rr33_hours[rr33_j]) * 3600,
                dtype=torch.float32
            )
        rr33_memories[(rr33_source, rr33_current)] = (
            rr33_mem, rr33_mask, rr33_age
        )

rr33_m, rr33_mask, rr33_age = rr33_memories[("world", False)]
rr33_n = rr33_m.shape[1]
print("rebuilt prior-memory RMS:",
      float((rr33_m - rr33_backup["core_probe_memory"][:8, :rr33_n])
            .square().mean().sqrt()))
print("rebuilt mask agreement:",
      bool(torch.equal(rr33_mask,
                       rr33_backup["core_probe_mask"][:8, :rr33_n])))

rr33_scores = []
try:
    for rr33_name in rr33_names:
        globals()[rr33_name] = rr33_backup[rr33_name][:8]
    mlp0 = rr33_mlp
    for (rr33_source, rr33_current), (rr33_mem, rr33_mask, rr33_age) in rr33_memories.items():
        for rr33_mem_norm in ("none", "plain", "block"):
            if rr33_mem_norm == "none":
                core_probe_memory = rr33_mem
            elif rr33_mem_norm == "plain":
                core_probe_memory = F.layer_norm(rr33_mem, (192,))
            else:
                core_probe_memory = norm0(rr33_mem, "core.blocks.0.0.norm2")
            core_probe_mask = rr33_mask
            for rr33_age_kind in ("none", "seconds_div10", "hours", "seconds"):
                core_probe_ages = rr33_age / 3600 if rr33_age_kind == "hours" else rr33_age
                rr33_scale = (
                    0.0 if rr33_age_kind == "none" else
                    0.1 if rr33_age_kind == "seconds_div10" else 1.0
                )
                for rr33_keys in (False, True):
                    for rr33_ff_kind in ("gelu", "relu", "silu"):
                        rr33_out, rr33_h = audit_core_update(
                            core_probe_z, rr33_scale, rr33_keys
                        )
                        rr33_scores.append((
                            float((rr33_out - rr33_target_out).square().mean().sqrt()),
                            float((rr33_h - rr33_target_h).square().mean().sqrt()),
                            rr33_source, rr33_current, rr33_mem_norm,
                            rr33_age_kind, rr33_keys, rr33_ff_kind
                        ))
    print("\n=== BEST MEMORY/ACTIVATION CONVENTIONS ===")
    for rr33_row in sorted(rr33_scores)[:12]:
        print(rr33_row)
finally:
    mlp0 = rr33_old_mlp
    for rr33_name, rr33_obj in rr33_backup.items():
        globals()[rr33_name] = rr33_obj

with open("/work/notes.md", "a") as rr33_f:
    rr33_f.write("\n## Run 33: memory convention audit\n"
                 "- Tested prior/current memory inclusion, memory sources and normalization, age units, and FFN activations.\n"
                 "- Best configurations: " + repr(sorted(rr33_scores)[:5]) + "\n"
                 "- No weights edited; approximate state agreement is not treated as validation.\n"
                 "- Python runs: 33; forward passes: 4.\n")
```