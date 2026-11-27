All 1,497 drug bundles now reconstruct accurately. However, forcing the existing hidden feature close to zero on factual records removes most of its dose responsiveness, so that feature alone is not a convincing repair. I’ll reconstruct the recurrent update to identify useful latent directions through forward-only tests. Close agreement with GPU states would validate that calculation.

```python
import itertools
import numpy as np
import torch
import torch.nn.functional as F

print("=== MODEL CONFIGURATION ===")
print(VOCAB["config"])

def core_lin(x, p):
    return F.linear(x, W0[p + ".weight"], W0.get(p + ".bias"))

def core_norm(x, p):
    return F.layer_norm(x, (192,), W0[p + ".weight"], W0[p + ".bias"], eps=1e-5)

def core_ffn(x, p):
    return core_lin(F.gelu(core_lin(x, p + ".0")), p + ".2")

def core_attn(x, p, nheads, memory=None, ages=None, valid=None, age_mode="both"):
    q = core_lin(x, p + ".q")
    if memory is None:
        kv = core_lin(x, p + ".kv")
        k, v = kv.chunk(2, dim=-1)
    else:
        k, v = core_lin(memory, p + ".kv").chunk(2, dim=-1)
        ka, va = core_lin(ages, p + ".kv").chunk(2, dim=-1)
        k = k + ka
        if age_mode == "both":
            v = v + va
    b, nq, width = q.shape
    nk = k.shape[1]
    q = q.reshape(b, nq, nheads, width // nheads).transpose(1, 2)
    k = k.reshape(b, nk, nheads, width // nheads).transpose(1, 2)
    v = v.reshape(b, nk, nheads, width // nheads).transpose(1, 2)
    attn_mask = None if valid is None else valid[:, None, None, :]
    ans = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    ans = ans.transpose(1, 2).reshape(b, nq, width)
    return core_lin(ans, p + ".out")

def core_layer(x, p, nheads, memory=None, ages=None, valid=None, age_mode="both"):
    x = x + core_attn(core_norm(x, p + ".norm1"), p + ".attn", nheads)
    if memory is not None:
        x = x + core_attn(core_norm(x, p + ".norm2"), p + ".cross", nheads,
                         memory, ages, valid, age_mode)
    return x + core_ffn(core_norm(x, p + ".norm3"), p + ".ffn")

core_state = torch.as_tensor(S0["x"], dtype=torch.float32)
core_e_all = recon_e.clone()
core_e_all[cf_rows] = cf_E
core_selected = np.flatnonzero(cf_frame.hour.to_numpy() >= 12)
core_selected = np.random.default_rng(1904).choice(
    core_selected, size=min(32, len(core_selected)), replace=False)
core_rows = cf_rows[core_selected]
core_previous = core_rows - 1
assert np.all(state_df.iloc[core_rows].stay_id.to_numpy() ==
              state_df.iloc[core_previous].stay_id.to_numpy())
core_dt = torch.tensor(
    (state_df.iloc[core_rows].hour.to_numpy() -
     state_df.iloc[core_previous].hour.to_numpy()) * 3600,
    dtype=torch.float32)
core_h_flow = core_state[core_previous, 192:960].reshape(-1, 4, 192) * torch.exp(
    -core_dt[:, None, None] * W0["core.flow.a"].exp()[None, None, :])
core_z = core_state[core_rows, -32:]
core_E_obs = cf_E[core_selected]
core_U_actions = cf_U[core_selected]
core_target_h = core_state[core_rows, 192:960].reshape(-1, 4, 192)
core_target_out = core_state[core_rows, :192]

def make_core_memory(include_current=False):
    histories = []
    for rr in core_rows:
        sid = state_df.iloc[rr].stay_id
        end = rr + int(include_current)
        hist = np.flatnonzero(
            (state_df.stay_id.to_numpy()[:end] == sid))[-128:]
        histories.append(hist)
    length = max(map(len, histories))
    memory = torch.zeros((len(histories), length, 192))
    age_seconds = torch.zeros((len(histories), length, 1))
    valid = torch.zeros((len(histories), length), dtype=torch.bool)
    for j, hist in enumerate(histories):
        memory[j, :len(hist)] = core_e_all[hist]
        age_seconds[j, :len(hist), 0] = torch.tensor(
            (state_df.iloc[core_rows[j]].hour -
             state_df.iloc[hist].hour.to_numpy()) * 3600, dtype=torch.float32)
        valid[j, :len(hist)] = True
    return memory, age_seconds, valid

core_memories = {c: make_core_memory(c) for c in [False, True]}

def core_update_local(z, config, h_flow=None, e=None, u=None, memory_data=None):
    nheads, age_divisor, age_mode, include_current = config
    h = core_h_flow if h_flow is None else h_flow
    e = core_E_obs if e is None else e
    u = core_U_actions if u is None else u
    memory, age_seconds, valid = (
        core_memories[include_current] if memory_data is None else memory_data)
    ages = core_ffn(torch.log1p(age_seconds.clamp_min(0)) / age_divisor, "core.age")
    tokens = torch.cat([h + W0["core.slots"][None, :, :],
                        e[:, None, :], u[:, None, :],
                        core_lin(z, "z_in")[:, None, :]], dim=1)
    tokens = core_layer(tokens, "core.prelude.0", nheads)
    E = core_norm(tokens[:, :4], "core.prelude_norm")
    base = h + E
    uu = base
    for p in ["core.blocks.0.0", "core.blocks.0.1"]:
        uu = core_layer(uu, p, nheads, memory, ages, valid, age_mode)
    delta = F.softplus(W0["core.delta_in"])
    abar = torch.exp(-delta * W0["core.a_in"].exp())
    hnew = abar * h + delta * core_lin(E, "core.b_in") + (uu - base)
    y = core_layer(core_lin(hnew, "core.c_out"), "core.coda.0", nheads)
    out = core_norm(y.mean(dim=1), "core.out_norm")
    return y, out

core_candidates = []
head_cfg = [v for k, v in VOCAB["config"].items()
            if "head" in k.lower() and isinstance(v, int) and v > 0 and 192 % v == 0]
head_choices = head_cfg[:1] if head_cfg else [4, 6, 8]
for cfg in itertools.product(head_choices, [1.0, 10.0], ["both", "key"], [False, True]):
    yh, yo = core_update_local(core_z, cfg)
    err_h = float(torch.sqrt(torch.mean((yh - core_target_h) ** 2)))
    err_out = float(torch.sqrt(torch.mean((yo - core_target_out) ** 2)))
    core_candidates.append((err_h + err_out, cfg, err_h, err_out))
core_candidates.sort(key=lambda x: x[0])
print("=== BEST RECURRENT RECONSTRUCTIONS ===")
for _, cfg, eh, eo in core_candidates[:8]:
    print({"config": cfg, "h_RMSE": round(eh, 6), "out_RMSE": round(eo, 6)})
core_best_cfg = core_candidates[0][1]

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 40: recurrent forward-only reconstruction\n"
            "Drug posterior exact on all 1497 rows (RMS 6.6e-5). Existing hidden contrast features are weak: insulin slope .31 at factual RMS .47; stronger zero constraints reduce slope. Reconstructed recurrent block and compared memory/age conventions against saved GPU states; diagnostics printed. No edits. Python runs: 40; forward passes: 3.\n")
```