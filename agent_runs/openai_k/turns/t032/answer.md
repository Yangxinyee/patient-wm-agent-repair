The posterior reconstruction is accurate enough for local experiments. I’ll now reconstruct the state update and test how latent changes reach the next reading. This could provide a targeted correction path. Agreement with saved states is required before trusting those sensitivity estimates; a large mismatch would rule them out.

```python
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

torch.set_num_threads(8)
torch.set_grad_enabled(False)

def audit_attention(x, memory, prefix, mask=None, key_memory=None):
    batch, nq, width = x.shape
    nk = memory.shape[1]
    qv = lin0(x, prefix + ".q")
    kv = lin0(memory, prefix + ".kv")
    kk, vv = kv.chunk(2, dim=-1)
    if key_memory is not None:
        kk = lin0(key_memory, prefix + ".kv")[..., :width]
    qv = qv.reshape(batch, nq, 4, 48).transpose(1, 2)
    kk = kk.reshape(batch, nk, 4, 48).transpose(1, 2)
    vv = vv.reshape(batch, nk, 4, 48).transpose(1, 2)
    logits = qv @ kk.transpose(-1, -2) / np.sqrt(48.0)
    if mask is not None:
        logits = logits.masked_fill(~mask[:, None, None, :], -1e30)
    av = torch.softmax(logits, dim=-1) @ vv
    av = av.transpose(1, 2).reshape(batch, nq, width)
    return lin0(av, prefix + ".out")

def audit_block(x, prefix, memory=None, mask=None, key_memory=None):
    normalized = norm0(x, prefix + ".norm1")
    x = x + audit_attention(normalized, normalized, prefix + ".attn")
    if memory is not None:
        x = x + audit_attention(
            norm0(x, prefix + ".norm2"), memory,
            prefix + ".cross", mask, key_memory
        )
    return x + mlp0(norm0(x, prefix + ".norm3"), prefix + ".ffn", "gelu")

r27_candidates = np.intersect1d(action_indices_audit, valid_previous_audit)
core_probe_indices = np.random.default_rng(271).choice(
    r27_candidates, size=64, replace=False
)
core_probe_previous = torch.as_tensor(
    sx[core_probe_indices - 1, 192:960].reshape(-1, 4, 192)
)
core_probe_dt = torch.as_tensor(
    dt_all_audit[core_probe_indices], dtype=torch.float32
)
core_probe_h = core_probe_previous * torch.exp(
    -core_probe_dt[:, None, None] * torch.exp(W0["core.flow.a"])[None, None, :]
)
core_probe_e = all_e_audit[core_probe_indices]
core_probe_u = all_u_audit[core_probe_indices]
core_probe_z = torch.as_tensor(sx[core_probe_indices, -32:])

r27_stay = np.asarray(train_states0["stay_id"])
r27_hour = np.asarray(train_states0["hour"])
r27_memory_lists = [
    np.flatnonzero((r27_stay == r27_stay[i]) & (np.arange(len(r27_stay)) < i))[-128:]
    for i in core_probe_indices
]
r27_max_memory = max(map(len, r27_memory_lists))
core_probe_memory = torch.zeros(64, r27_max_memory, 192)
core_probe_mask = torch.zeros(64, r27_max_memory, dtype=torch.bool)
core_probe_ages = torch.zeros(64, r27_max_memory, 1)
for r27_j, r27_mem in enumerate(r27_memory_lists):
    core_probe_memory[r27_j, :len(r27_mem)] = all_e_audit[r27_mem]
    core_probe_mask[r27_j, :len(r27_mem)] = True
    core_probe_ages[r27_j, :len(r27_mem), 0] = torch.as_tensor(
        np.maximum(0, (r27_hour[core_probe_indices[r27_j]] - r27_hour[r27_mem]) * 3600),
        dtype=torch.float32
    )

def audit_core_update(z, age_scale=1.0, keys_only=False):
    h = core_probe_h
    tokens = torch.cat([
        h + W0["core.slots"][None, :, :],
        core_probe_e[:, None, :],
        core_probe_u[:, None, :],
        lin0(z, "z_in")[:, None, :]
    ], dim=1)
    tokens = audit_block(tokens, "core.prelude.0")
    E = norm0(tokens[:, :4], "core.prelude_norm")
    age = mlp0(torch.log1p(core_probe_ages) * age_scale, "core.age", "gelu")
    key_memory = core_probe_memory + age
    value_memory = core_probe_memory if keys_only else key_memory
    state = h + E
    state = audit_block(
        state, "core.blocks.0.0", value_memory, core_probe_mask,
        key_memory if keys_only else None
    )
    state = audit_block(
        state, "core.blocks.0.1", value_memory, core_probe_mask,
        key_memory if keys_only else None
    )
    delta = F.softplus(W0["core.delta_in"])
    retention = torch.exp(-delta * torch.exp(W0["core.a_in"]))
    new_h = retention[None, None, :] * h
    new_h = new_h + delta[None, None, :] * lin0(E, "core.b_in")
    new_h = new_h + state - (h + E)
    new_h = audit_block(lin0(new_h, "core.c_out"), "core.coda.0")
    out = norm0(new_h.mean(dim=1), "core.out_norm")
    return out, new_h

r27_target_out = torch.as_tensor(sx[core_probe_indices, :192])
r27_target_h = torch.as_tensor(
    sx[core_probe_indices, 192:960].reshape(-1, 4, 192)
)
core_reconstruction_results = {}
print("=== CORE CONVENTION CHECK ===")
for r27_scale in [1.0, 0.1]:
    for r27_keys_only in [False, True]:
        r27_out, r27_h = audit_core_update(
            core_probe_z, r27_scale, r27_keys_only
        )
        r27_out_error = float((r27_out - r27_target_out).square().mean().sqrt())
        r27_h_error = float((r27_h - r27_target_h).square().mean().sqrt())
        core_reconstruction_results[(r27_scale, r27_keys_only)] = (
            r27_out_error, r27_h_error
        )
        print("age scale", r27_scale, "keys only", r27_keys_only,
              "out RMS", r27_out_error, "h RMS", r27_h_error)

print("\n=== HEAD TENSORS ===")
for r27_name in NAMES:
    if r27_name.startswith((
        "query", "type_emb", "last_", "embed_transition",
        "attr_heads.value", "monitor"
    )):
        print(r27_name, SHAPES[r27_name])
print("latent mean SD:", np.std(sx[:, -32:], axis=0))
print("training IDs sample:", ids[:3], ids[100:103])

with Path("/work/notes.md").open("a") as r27_file:
    r27_file.write(
        "\n## Run 27: local core reconstruction\n"
        "- Full posterior reconstruction passed in run 25; its later grouping "
        "print failed because bundle entries are (record, kind) pairs.\n"
        f"- Core convention results (out/h RMS): {core_reconstruction_results}.\n"
        "- Testing a conditionally centered dose-residual correction in "
        "posterior means; no edit until the transfer path is validated.\n"
        "- Python runs: 27; forward passes: 4; no weights edited.\n"
    )
```