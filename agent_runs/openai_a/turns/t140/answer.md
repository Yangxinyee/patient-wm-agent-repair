The measurement encoding matches the saved posterior to about \(1.7\times10^{-6}\). Bundle counts also show infusion-end atoms that are absent as separate CSV rows, so memory reconstruction must respect the saved bundle timeline. I’ll now test complete transitions under the plausible memory conventions. Only a numerical-precision match will support fitting synthetic-dose contrasts.

```python
import time, itertools
import torch.nn.functional as F
t124 = time.time()

rec_code_ids = {c["name"]: c["id"] for c in VOCAB["codes"]}
rec_bundle_index = {(int(s), float(h)): i for i, (s, h) in
                    enumerate(zip(rm_states["stay_id"], rm_states["hour"]))}
rec_world_rows, rec_world_ix = [], []
for (sid, hr, ev), row in rm_lookup.items():
    if ev in ("insulin", "norepinephrine"):
        continue
    ix = rec_bundle_index.get((int(sid), float(hr)))
    if ix is not None:
        rec_world_rows.append(row)
        rec_world_ix.append(ix)

rec_atom_count = len(rec_world_rows)
rec_atom = torch.zeros((rec_atom_count, 192))
rec_numbers = torch.zeros((rec_atom_count, 6))
for j, row in enumerate(rec_world_rows):
    ev = row.event
    ek = 1 if ev in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff") else (
        3 if ev == "hospital_admission" else 4)
    unit = 2 if ev in ("map_arterial", "map_cuff") else 0
    rec_atom[j] = (rm_W["embed.code.weight"][rec_code_ids[ev]]
                   + rm_W["embed.kind.weight"][0]
                   + rm_W["embed.ekind.weight"][ek]
                   + rm_W["embed.kq.weight"][0]
                   + rm_W["embed.categorical.unit.weight"][unit])
    if ek == 1:
        ns = rm_normspec[(ev, "value")]
        v = float(row.value)
        rec_numbers[j, 0] = np.clip((v - ns["median"]) / ns["iqr"], -10, 10)
        rec_numbers[j, 1] = 1
        bi = int(np.searchsorted(ns["fine_bin_edges"], v, side="left")) + 1
        rec_atom[j] += rm_W["embed.bins.value.weight"][bi]
rec_atom += rm_lin(rec_numbers, "embed.numeric")
rec_world_ix_t = torch.tensor(rec_world_ix, dtype=torch.long)
rec_counts = torch.bincount(rec_world_ix_t, minlength=len(rm_states["x"]))
rec_sum = torch.zeros((len(rec_counts), 192))
rec_sum.index_add_(0, rec_world_ix_t, rm_mlp(rec_atom, "encode.mlp1"))
rec_e_all = rm_ln(rm_mlp(
    rec_sum / rec_counts.clamp_min(1).sqrt()[:, None]
    + rm_lin(torch.log1p(rec_counts.float())[:, None], "encode.size"),
    "encode.mlp2"), "encode.norm")

def rec_attention(query, key, value, prefix):
    b, nq, _ = query.shape
    nk = key.shape[1]
    q = rm_lin(query, prefix + ".q").reshape(b, nq, 4, 48).transpose(1, 2)
    kw = rm_W[prefix + ".kv.weight"][:192]
    vw = rm_W[prefix + ".kv.weight"][192:]
    k = F.linear(key, kw).reshape(b, nk, 4, 48).transpose(1, 2)
    v = F.linear(value, vw).reshape(b, nk, 4, 48).transpose(1, 2)
    return q, k, v

def rec_attn_apply(x, prefix, memory=None, ages=None, mask=None,
                   age_mode="keys_input"):
    qx = rm_ln(x, prefix + (".norm1" if memory is None else ".norm2"))
    ap = prefix + (".attn" if memory is None else ".cross")
    if memory is None:
        q, k, v = rec_attention(qx, qx, qx, ap)
    else:
        key = memory + ages if age_mode in ("keys_input", "both_input") else memory
        val = memory + ages if age_mode == "both_input" else memory
        q, k, v = rec_attention(qx, key, val, ap)
        if age_mode == "keys_output":
            k += ages.reshape(len(x), -1, 4, 48).transpose(1, 2)
    score = q @ k.transpose(-1, -2) / np.sqrt(48)
    if mask is not None:
        score = score.masked_fill(~mask[:, None, None, :], -1e9)
    y = (score.softmax(-1) @ v).transpose(1, 2).reshape(len(x), x.shape[1], 192)
    return x + rm_lin(y, ap + ".out")

def rec_layer(x, prefix, memory=None, ages=None, mask=None, age_mode="keys_input"):
    x = rec_attn_apply(x, prefix)
    if memory is not None:
        x = rec_attn_apply(x, prefix, memory, ages, mask, age_mode)
    return x + rm_mlp(rm_ln(x, prefix + ".norm3"), prefix + ".ffn")

rec_hf = rec_context(rm_prev, rm_dt)[1]
rec_tokens = torch.cat([
    rec_hf + rm_W["core.slots"][None],
    r96_best["e"][:, None], r96_best["u"][:, None],
    rm_lin(rm_target_z, rec_zin)[:, None]], 1)
rec_E = rm_ln(rec_layer(rec_tokens, "core.prelude.0")[:, :4], "core.prelude_norm")
rec_hbase = rec_hf + rec_E
rec_delta = F.softplus(rm_W["core.delta_in"])
rec_abar = torch.exp(-rec_delta * torch.exp(rm_W["core.a_in"]))
rec_h_target = torch.as_tensor(rm_states["x"][rm_pure_idx, 192:960]).reshape(-1, 4, 192)
rec_out_target = torch.as_tensor(rm_states["x"][rm_pure_idx, :192])

rec_transition_checks = []
for world_only, include_current in itertools.product([False, True], [False, True]):
    selections = []
    for idx in rm_pure_idx:
        sid = rm_states["stay_id"][idx]
        jx = np.flatnonzero((rm_states["stay_id"] == sid)
                           & (np.arange(len(rec_counts)) < idx + int(include_current)))
        if world_only:
            jx = jx[rec_counts[jx].numpy() > 0]
        selections.append(jx[-128:])
    width = max(map(len, selections))
    mem = torch.zeros((len(selections), width, 192))
    age_seconds = torch.zeros((len(selections), width, 1))
    valid_mem = torch.zeros((len(selections), width), dtype=torch.bool)
    for j, jx in enumerate(selections):
        mem[j, :len(jx)] = rec_e_all[jx]
        age_seconds[j, :len(jx), 0] = torch.tensor(
            (rm_states["hour"][rm_pure_idx[j]] - rm_states["hour"][jx]) * 3600,
            dtype=torch.float32)
        valid_mem[j, :len(jx)] = True
    for age_clock, age_div, mode in itertools.product(
            [1.0, 3600.0], [1.0, 10.0],
            ["keys_input", "both_input", "keys_output", "none"]):
        ag = rm_mlp(torch.log1p(age_seconds / age_clock) / age_div, "core.age")
        u = rec_hbase
        for layer_id in range(2):
            u = rec_layer(u, "core.blocks.0." + str(layer_id), mem, ag, valid_mem, mode)
        hn = rec_abar * rec_hf + rec_delta * rm_lin(rec_E, "core.b_in") + u - rec_hbase
        hn = rec_layer(rm_lin(hn, "core.c_out"), "core.coda.0")
        on = rm_ln(hn.mean(1), rec_outnorm)
        he = float((hn - rec_h_target).square().mean().sqrt())
        oe = float((on - rec_out_target).square().mean().sqrt())
        spec = (world_only, include_current, age_clock, age_div, mode)
        rec_transition_checks.append((he, oe, spec))
rec_transition_checks.sort()
print("BEST COMPLETE-TRANSITION CHECKS:", rec_transition_checks[:10])
print("TARGET SLOT RMS:", float(rec_h_target.square().mean().sqrt()))
print("ENCODED WORLD BUNDLES:", int((rec_counts > 0).sum()))
print("SECONDS:", round(time.time() - t124, 2))
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Complete-transition reconstruction; Python 124\n"
            "- Measurement encoding verified: event kind 1, glucose unit 0, MAP unit 2, kq 0.\n"
            "- Testing full recurrent update and memory conventions against saved GPU states.\n"
            "- No new weight edits; synthetic contrasts require a successful state match.\n")
```