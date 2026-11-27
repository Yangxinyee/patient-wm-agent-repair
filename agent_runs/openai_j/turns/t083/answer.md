The saved helpers expose the required embeddings, posterior means, and readout features. I’ll now test a CPU reconstruction of the recurrent update on action bundles, including plausible memory-age conventions. If one convention closely reproduces the saved states, we can use it to measure dose sensitivity downstream of the posterior.

```python
import torch.nn.functional as Fnn
torch.set_num_threads(8)

def repair_self_attention(x, prefix):
    n, length, width = x.shape
    q = Fnn.linear(x, wt(prefix + ".q.weight")).reshape(n, length, 4, 48).transpose(1, 2)
    kv = Fnn.linear(x, wt(prefix + ".kv.weight"))
    k, v = kv.chunk(2, dim=-1)
    k = k.reshape(n, length, 4, 48).transpose(1, 2)
    v = v.reshape(n, length, 4, 48).transpose(1, 2)
    a = torch.softmax(q @ k.transpose(-1, -2) / np.sqrt(48), dim=-1)
    y = (a @ v).transpose(1, 2).reshape(n, length, width)
    return Fnn.linear(y, wt(prefix + ".out.weight"))

def repair_simple_block(x, prefix):
    x = x + repair_self_attention(ln(x, prefix + ".norm1"), prefix + ".attn")
    return x + mlp(ln(x, prefix + ".norm3"), prefix + ".ffn", "gelu")

def repair_cross_attention(x, memory, ages, memory_mask, prefix,
                           age_scale=10.0, age_mode="outputkeys"):
    n, length, width = x.shape
    mlength = memory.shape[1]
    age_vector = mlp(torch.log1p(ages).unsqueeze(-1) / age_scale, "core.age", "gelu")
    if age_mode == "inputboth":
        k, v = Fnn.linear(memory + age_vector, wt(prefix + ".kv.weight")).chunk(2, -1)
    else:
        k, v = Fnn.linear(memory, wt(prefix + ".kv.weight")).chunk(2, -1)
        if age_mode == "outputkeys":
            k = k + age_vector
        elif age_mode == "inputkeys":
            k = k + Fnn.linear(age_vector, wt(prefix + ".kv.weight")[:192])
    q = Fnn.linear(x, wt(prefix + ".q.weight")).reshape(n, length, 4, 48).transpose(1, 2)
    k = k.reshape(n, mlength, 4, 48).transpose(1, 2)
    v = v.reshape(n, mlength, 4, 48).transpose(1, 2)
    scores = q @ k.transpose(-1, -2) / np.sqrt(48)
    scores = scores.masked_fill(~memory_mask[:, None, None, :], -1e9)
    y = (torch.softmax(scores, -1) @ v).transpose(1, 2).reshape(n, length, width)
    return Fnn.linear(y, wt(prefix + ".out.weight"))

def repair_core_step(h, e, u, z, memory, ages, memory_mask,
                     age_scale=10.0, age_mode="outputkeys"):
    z_token = Fnn.linear(z, wt("z_in.weight"), wt("z_in.bias"))
    tokens = torch.cat([h + wt("core.slots")[None], e[:, None],
                        u[:, None], z_token[:, None]], dim=1)
    prelude = repair_simple_block(tokens, "core.prelude.0")
    E_local = ln(prelude[:, :4], "core.prelude_norm")
    recurrent = h + E_local
    for prefix in ["core.blocks.0.0", "core.blocks.0.1"]:
        recurrent = recurrent + repair_self_attention(ln(recurrent, prefix + ".norm1"),
                                                       prefix + ".attn")
        recurrent = recurrent + repair_cross_attention(
            ln(recurrent, prefix + ".norm2"), memory, ages, memory_mask,
            prefix + ".cross", age_scale, age_mode)
        recurrent = recurrent + mlp(ln(recurrent, prefix + ".norm3"),
                                    prefix + ".ffn", "gelu")
    delta = Fnn.softplus(wt("core.delta_in"))
    abar = torch.exp(-delta * torch.exp(wt("core.a_in")))
    next_h = (abar * h + delta * Fnn.linear(E_local, wt("core.b_in.weight"))
              + recurrent - (h + E_local))
    next_h = Fnn.linear(next_h, wt("core.c_out.weight"))
    next_h = repair_simple_block(next_h, "core.coda.0")
    next_out = ln(next_h.mean(1), "core.out_norm")
    return next_out, next_h

core_test_rows = np.flatnonzero(
    (post_rates.sum(1) > 0) & post_valid &
    np.r_[False, SSID[1:] == SSID[:-1]]
)[::25][:40]
core_prev = core_test_rows - 1
core_dt = torch.tensor((STIME[core_test_rows] - STIME[core_prev]) * 3600,
                       dtype=torch.float32)
core_h = torch.tensor(SX[core_prev, 192:960]).reshape(-1, 4, 192)
core_h = core_h * torch.exp(-core_dt[:, None, None] * torch.exp(wt("core.flow.a")))
core_e = post_e0[core_test_rows]
core_u = post_u0[core_test_rows]
core_z = torch.tensor(SX[core_test_rows, 960:992])
core_truth_h = torch.tensor(SX[core_test_rows, 192:960]).reshape(-1, 4, 192)
core_truth_out = torch.tensor(SX[core_test_rows, :192])

observation_counts = np.bincount(
    post_atom_bundle[~post_action_mask], minlength=len(SX))
core_checks = []
for observation_only in [False, True]:
    for include_current in [False, True]:
        memory_indices = []
        for row in core_test_rows:
            candidates = np.flatnonzero(
                (SSID == SSID[row]) &
                (np.arange(len(SX)) <= row if include_current
                 else np.arange(len(SX)) < row))
            if observation_only:
                candidates = candidates[observation_counts[candidates] > 0]
            memory_indices.append(candidates[-128:])
        max_memory = max(map(len, memory_indices))
        memory = torch.zeros(len(core_test_rows), max_memory, 192)
        ages = torch.zeros(len(core_test_rows), max_memory)
        memory_mask = torch.zeros(len(core_test_rows), max_memory, dtype=torch.bool)
        for j, indices in enumerate(memory_indices):
            memory[j, :len(indices)] = post_e0[indices]
            ages[j, :len(indices)] = torch.tensor(
                (STIME[core_test_rows[j]] - STIME[indices]) * 3600,
                dtype=torch.float32)
            memory_mask[j, :len(indices)] = True
        for age_scale in [1.0, 10.0]:
            for age_mode in ["outputkeys", "inputkeys", "inputboth"]:
                predicted_out, predicted_h = repair_core_step(
                    core_h, core_e, core_u, core_z, memory, ages, memory_mask,
                    age_scale, age_mode)
                core_checks.append({
                    "observation_only": observation_only,
                    "include_current": include_current,
                    "age_scale": age_scale, "age_mode": age_mode,
                    "out_rmse": float(torch.mean((predicted_out-core_truth_out)**2).sqrt()),
                    "h_rmse": float(torch.mean((predicted_h-core_truth_h)**2).sqrt())
                })
core_check_table = pd.DataFrame(core_checks).sort_values("out_rmse")
print("TEST ACTION BUNDLES:", len(core_test_rows))
print(core_check_table.head(12).to_string(index=False))
print("ACTIVE EDITS:", edited())
```