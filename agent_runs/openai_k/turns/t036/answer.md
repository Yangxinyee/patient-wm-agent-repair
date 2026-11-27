The mismatch could come from attention or memory-age conventions rather than the drug encoding: the posterior already reconstructs closely. I’ll test those conventions against saved states without editing weights. A near-zero error would identify the computation; a merely smaller error would not yet validate it.

```python
import torch
import torch.nn.functional as F
import numpy as np

print("=== CONFIGURATION ===")
print(VOCAB.get("config", {}))
print("memory mask counts:", core_probe_mask.sum(1)[:12].tolist())
print("probe hours:", [state_keys_audit[int(i)] for i in core_probe_indices[:6]])

rr31_old_attention = audit_attention
rr31_heads = 4
rr31_mask_valid = True

def rr31_attention(x, memory, prefix, mask=None, key_memory=None):
    bs, nq, wd = x.shape
    nk = memory.shape[1]
    key_source = memory if key_memory is None else key_memory
    q = lin0(x, prefix + ".q")
    k = lin0(key_source, prefix + ".kv")[..., :wd]
    v = lin0(memory, prefix + ".kv")[..., wd:]
    hd = wd // rr31_heads
    q = q.reshape(bs, nq, rr31_heads, hd).transpose(1, 2)
    k = k.reshape(bs, nk, rr31_heads, hd).transpose(1, 2)
    v = v.reshape(bs, nk, rr31_heads, hd).transpose(1, 2)
    logits = (q @ k.transpose(-1, -2)) / np.sqrt(float(hd))
    if mask is not None:
        bad = ~mask.bool() if rr31_mask_valid else mask.bool()
        logits = logits.masked_fill(bad[:, None, None, :], -1e30)
    av = torch.softmax(logits, dim=-1) @ v
    av = av.transpose(1, 2).reshape(bs, nq, wd)
    return lin0(av, prefix + ".out")

rr31_x = norm0(core_probe_h, "core.blocks.0.0.norm2")
rr31_ref = rr31_old_attention(
    rr31_x, core_probe_memory, "core.blocks.0.0.cross",
    mask=core_probe_mask
)
for rr31_valid in (True, False):
    rr31_mask_valid = rr31_valid
    rr31_new = rr31_attention(
        rr31_x, core_probe_memory, "core.blocks.0.0.cross",
        mask=core_probe_mask
    )
    print("replacement attention RMS; True means valid =", rr31_valid,
          float(torch.mean((rr31_new - rr31_ref)**2).sqrt()))
rr31_mask_valid = min(
    (True, False),
    key=lambda valid: (
        globals().__setitem__("rr31_mask_valid", valid),
        float(torch.mean((rr31_attention(
            rr31_x, core_probe_memory, "core.blocks.0.0.cross",
            mask=core_probe_mask
        ) - rr31_ref)**2))
    )[1]
)
print("chosen mask convention:", rr31_mask_valid)

rr31_probe_names = [
    "core_probe_h", "core_probe_e", "core_probe_u", "core_probe_z",
    "core_probe_memory", "core_probe_mask", "core_probe_ages"
]
rr31_backups = {n: globals()[n] for n in rr31_probe_names}
rr31_idx = np.asarray(core_probe_indices[:16], dtype=int)
rr31_target = torch.as_tensor(sx[rr31_idx], dtype=torch.float32)
rr31_target_out = rr31_target[:, :192]
rr31_target_h = rr31_target[:, 192:960].reshape(-1, 4, 192)
rr31_scores = []

try:
    for n in rr31_probe_names:
        globals()[n] = rr31_backups[n][:16]
    audit_attention = rr31_attention
    for rr31_heads in (1, 2, 3, 4, 6, 8, 12, 16):
        for rr31_age_scale in (0.0, 0.05, 0.1, 0.2, 1.0):
            for rr31_keys in (False, True):
                rr31_result = audit_core_update(
                    core_probe_z, age_scale=rr31_age_scale,
                    keys_only=rr31_keys
                )
                rr31_out, rr31_h = rr31_result
                if rr31_out.ndim == 3:
                    rr31_h, rr31_out = rr31_out, rr31_h
                rr31_out_error = float(((rr31_out - rr31_target_out)**2).mean().sqrt())
                rr31_h_error = float(((rr31_h - rr31_target_h)**2).mean().sqrt())
                rr31_scores.append((rr31_out_error, rr31_h_error,
                                    rr31_heads, rr31_age_scale, rr31_keys))
    print("\n=== BEST RECURRENT CONVENTIONS ===")
    for rr31_row in sorted(rr31_scores)[:12]:
        print(rr31_row)
finally:
    audit_attention = rr31_old_attention
    for n, obj in rr31_backups.items():
        globals()[n] = obj

with open("/work/notes.md", "a") as rr31_f:
    rr31_f.write("\n## Run 31: attention and memory-age convention search\n"
                 "- Tested attention head counts and age scaling against saved states; no model weights changed.\n"
                 "- Best configurations: " + repr(sorted(rr31_scores)[:5]) + "\n"
                 "- Python runs: 31; forward passes: 4; no weights edited.\n")
```