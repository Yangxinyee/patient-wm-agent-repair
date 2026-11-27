The state layout is correct to numerical precision. The official reached-hour shifts are −3.680, +2.018, +0.465, and −0.871, with factual NLL 1.63164. I’ll now test the head on readings with no previous same-code value, eliminating register-order ambiguity. Error near the estimated sampling noise would validate the head settings.

```python
import itertools, time
torch.set_num_threads(8)

r30_W = {}
def r30_w(name):
    if name not in r30_W:
        r30_W[name] = torch.as_tensor(load_original(name), dtype=torch.float32)
    return r30_W[name]

def r30_lin(x, name):
    return torch.nn.functional.linear(x, r30_w(name+".weight"), r30_w(name+".bias"))

def r30_norm(x, name):
    return torch.nn.functional.layer_norm(
        x, (192,), r30_w(name+".weight"), r30_w(name+".bias"))

def r30_act(x, name):
    return {"gelu": torch.nn.functional.gelu,
            "silu": torch.nn.functional.silu,
            "relu": torch.nn.functional.relu}[name](x)

def r30_mlp(x, name, activation):
    return r30_lin(r30_act(r30_lin(x, name+".0"), activation), name+".2")

r30_sel = np.flatnonzero(has22 == 0)[:120]
r30_pre = torch.tensor(r29_export["x"][pre22[r30_sel]])
r30_out = r30_pre[:, :192]
r30_h = r30_pre[:, 192:960].reshape(-1, 4, 192)
r30_dt = torch.tensor(dt22[r30_sel], dtype=torch.float32)
r30_flow = torch.exp(-r30_dt[:, None, None] *
                     torch.exp(r30_w("core.flow.a"))[None, None, :])
r30_base = r30_out + r30_norm((r30_h*r30_flow).mean(1), "core.out_norm")
r30_code = code22[r30_sel]
r30_centres = centres22[r30_sel]
r30_transition = transition22[r30_sel]
r30_ref = torch.tensor(reference_mean22[r30_sel], dtype=torch.float32)
r30_nll_ref = torch.tensor(reference_nll22[r30_sel], dtype=torch.float32)
r30_bin = observed_bin22[r30_sel]
r30_code_hidden = torch.nn.functional.linear(
    r30_w("embed.code.weight")[r30_code],
    r30_w("attr_heads.value_bin.0.weight"))
r30_type = r30_w("type_emb.weight")
r30_constants = []
for r30_nb, r30_vb, r30_rb, r30_st in itertools.product(
        (0, 1), (0, 1), (0, 1), (-1, 0, 1)):
    r30_c = torch.zeros(192)
    if r30_nb:
        r30_c += r30_w("last_numeric.bias")
    if r30_vb:
        r30_c += r30_w("embed.bins.value.weight")[0]
    if r30_rb:
        r30_c += r30_w("embed.bins.rate.weight")[0]
    if r30_st >= 0:
        r30_c += r30_w("last_status.weight")[r30_st]
    r30_constants.append(((r30_nb, r30_vb, r30_rb, r30_st),
        torch.nn.functional.linear(r30_c, r30_w("attr_heads.value_bin.0.weight"))))

print("No-last sample size:", len(r30_sel))
r30_results = []
r30_start = time.time()
for r30_ga, r30_unit, r30_ps, r30_vs, r30_pa in itertools.product(
        ("gelu", "silu", "relu"), ("seconds", "hours"),
        (0.1, 1.0), (0.1, 1.0), ("gelu", "silu", "relu")):
    r30_logdt = torch.log1p(r30_dt / (3600.0 if r30_unit == "hours" else 1.0))
    r30_gp = r30_mlp((r30_logdt*r30_ps)[:, None], "gap", r30_ga)
    r30_gv = r30_mlp((r30_logdt*r30_vs)[:, None], "gap", r30_ga)
    r30_raw = r30_mlp(r30_base+r30_gp, "z_prior", r30_pa)
    r30_view = r30_base+r30_gv+r30_lin(r30_raw[:, :32], "z_head")
    r30_q = r30_norm(r30_view[None, :, :]+r30_type[:, None, :], "query_norm")
    r30_first = r30_lin(r30_q, "attr_heads.value_bin.0")+r30_code_hidden[None, :, :]
    for r30_constspec, r30_ch in r30_constants:
        for r30_ha in ("gelu", "silu", "relu"):
            r30_hidden = r30_act(r30_first+r30_ch[None, None, :], r30_ha)
            r30_logits = r30_lin(r30_hidden, "attr_heads.value_bin.2")+r30_transition[None, :, :]
            r30_prob = r30_logits.softmax(-1)
            r30_mean = (r30_prob[:, :, 1:]*r30_centres[None, :, :]).sum(-1)
            r30_errors = (r30_mean-r30_ref[None, :]).square().mean(1).sqrt()
            r30_ti = int(r30_errors.argmin())
            r30_nll = -torch.log(r30_prob[r30_ti].gather(
                1, r30_bin[:, None]).squeeze(1).clamp_min(1e-12))
            r30_results.append((
                float(r30_errors[r30_ti]),
                float((r30_nll-r30_nll_ref).abs().mean()),
                r30_ga, r30_unit, r30_ps, r30_vs, r30_pa,
                r30_ha, r30_ti, r30_constspec))
r30_results.sort()
print("Elapsed seconds:", round(time.time()-r30_start, 1))
print("Best deterministic no-last specifications:")
for r30_result in r30_results[:12]:
    print(r30_result)

r30_mc_results = []
r30_generator = torch.Generator().manual_seed(3030)
r30_eps = torch.randn(64, len(r30_sel), 32, generator=r30_generator)
for r30_result in r30_results[:8]:
    _, _, r30_ga, r30_unit, r30_ps, r30_vs, r30_pa, r30_ha, r30_ti, r30_cs = r30_result
    r30_logdt = torch.log1p(r30_dt/(3600.0 if r30_unit == "hours" else 1.0))
    r30_raw = r30_mlp(r30_base+r30_mlp(
        (r30_logdt*r30_ps)[:, None], "gap", r30_ga), "z_prior", r30_pa)
    r30_z = r30_raw[None, :, :32]+r30_eps*torch.exp(
        r30_raw[None, :, 32:].clamp(-7, 3))
    r30_view = r30_base[None, :, :]+r30_mlp(
        (r30_logdt*r30_vs)[:, None], "gap", r30_ga)[None, :, :]+r30_lin(r30_z, "z_head")
    r30_q = r30_norm(r30_view+r30_type[r30_ti], "query_norm")
    r30_ch = dict(r30_constants)[r30_cs]
    r30_first = r30_lin(r30_q, "attr_heads.value_bin.0")+r30_code_hidden[None, :, :]+r30_ch
    r30_p = (r30_lin(r30_act(r30_first, r30_ha), "attr_heads.value_bin.2")
             +r30_transition[None, :, :]).softmax(-1)
    r30_m = (r30_p[:, :, 1:]*r30_centres[None, :, :]).sum(-1)
    r30_pbar = r30_p.mean(0)
    r30_nll = -torch.log(r30_pbar.gather(1, r30_bin[:, None]).squeeze(1).clamp_min(1e-12))
    r30_mc_results.append((
        float((r30_m.mean(0)-r30_ref).square().mean().sqrt()),
        float((r30_m.var(0, unbiased=False).mean()/4).sqrt()),
        float((r30_nll-r30_nll_ref).abs().mean()),
        r30_result[2:]))
print("MC checks: RMSE, expected four-sample noise, NLL MAE, specification")
for r30_result in sorted(r30_mc_results):
    print(r30_result)

with open("/work/notes.md", "a") as r30_note:
    r30_note.write(
        "\n## Run 30\n"
        "State export identity verified (RMSE 7.4e-8). Official original reached-row "
        "shifts: I x2 -3.680, half +2.018; NE x2 +0.465, half -0.871. Original "
        "factual NLL 1.63164. Separately testing gap/prior/head activations and "
        "absent-register constants on no-last cases; results retained in "
        "r30_results and r30_mc_results. No edits.\n"
    )
```