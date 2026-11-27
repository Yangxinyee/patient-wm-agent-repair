The saved states are aligned correctly, and the status embedding is zero for readings. That rules out two suspected causes. The architecture distinguishes the gap used by the prior from the gap used by the reading view; I’ll test those separately, along with the presence flag. A sharp reduction in error would support a convention mismatch rather than sampling noise.

```python
import itertools
import torch.nn.functional as F
torch.set_num_threads(8)

def rc_lin(x, name):
    return F.linear(x, W0[name + ".weight"], W0.get(name + ".bias"))

def rc_norm(x, name):
    return F.layer_norm(x, (192,), W0[name + ".weight"], W0[name + ".bias"], eps=1e-5)

def rc_act(x, kind):
    return F.silu(x) if kind == "silu" else F.gelu(x)

def rc_mlp(x, name, kind="silu"):
    return rc_lin(rc_act(rc_lin(x, name + ".0"), kind), name + ".2")

rc_n = len(hm)
rc_has = torch.tensor(hm["has_last"].to_numpy(), dtype=torch.float32)
rc_age = torch.tensor(hm["age"].to_numpy(), dtype=torch.float32)
rc_z = torch.tensor(hm["last_z"].to_numpy(), dtype=torch.float32)
rc_bins = torch.tensor(hm["last_bin"].to_numpy(), dtype=torch.long)
rc_codes = torch.tensor([code_ids[e] for e in hm["event"]], dtype=torch.long)
rc_centres = centres.clone()
rc_dt = dt0.clone()
rc_base = out0 + rc_norm(
    (h0 * torch.exp(-rc_dt[:, None] * W0["core.flow.a"].exp()[None, None])).mean(1),
    "core.out_norm"
)
rc_gap_inputs = {
    "seconds_div10": torch.log1p(rc_dt) / 10,
    "seconds": torch.log1p(rc_dt),
    "hours_div10": torch.log1p(rc_dt / 3600) / 10,
    "hours": torch.log1p(rc_dt / 3600),
}
rc_gaps = {(mode, act): rc_mlp(inp, "gap", act)
           for mode, inp in rc_gap_inputs.items() for act in ("silu", "gelu")}
rc_regs = {}
for age_mode, flag_col in itertools.product(("seconds", "hours"), (None, 3, 4, 5, 6, 7)):
    ff0 = torch.zeros(rc_n, 8)
    ff0[:, 0] = rc_has
    ff0[:, 1] = torch.log1p(rc_age / (3600 if age_mode == "hours" else 1)) / 10
    ff0[:, 2] = rc_z
    if flag_col is not None:
        ff0[:, flag_col] = rc_has
    rc_regs[(age_mode, flag_col)] = (
        rc_lin(ff0, "last_numeric") + W0["embed.bins.value.weight"][rc_bins]
        + W0["embed.bins.rate.weight"][0] + W0["last_status.weight"][0]
    )

rc_search = []
for prior_act, head_act in itertools.product(("silu", "gelu"), repeat=2):
    for ctx_gap, view_gap in itertools.product(rc_gap_inputs, repeat=2):
        # First search uses SiLU for the gap MLP; its activation is tested below.
        ctx0 = rc_base + rc_gaps[(ctx_gap, "silu")]
        zp0 = rc_mlp(ctx0, "z_prior", prior_act)
        view0 = rc_base + rc_gaps[(view_gap, "silu")] + rc_lin(zp0[:, :32], "z_head")
        qq0 = rc_norm(view0 + W0["type_emb.weight"][1], "query_norm")
        for reg_cfg, reg0 in rc_regs.items():
            wc0 = qq0 + W0["embed.code.weight"][rc_codes] + reg0
            logits0 = rc_mlp(wc0, "attr_heads.value_bin", head_act)
            logits0 += W0["embed_transition.value_bin.weight"][rc_codes * 17 + rc_bins]
            mm0 = (logits0.softmax(-1) * rc_centres).sum(-1).numpy()
            rmse0 = float(np.sqrt(np.mean((mm0 - target_means)**2)))
            rc_search.append((rmse0, (prior_act, head_act, ctx_gap, view_gap, *reg_cfg)))

rc_search.sort(key=lambda x: x[0])
print("Best deterministic configurations:")
for row in rc_search[:12]:
    print(round(row[0], 4), row[1])

rc_mc_results = []
torch.manual_seed(2901)
rc_noise = torch.randn(48, rc_n, 32)
for _, cfg in rc_search[:6]:
    prior_act, head_act, ctx_gap, view_gap, age_mode, flag_col = cfg
    for gap_act in ("silu", "gelu"):
        ctx0 = rc_base + rc_gaps[(ctx_gap, gap_act)]
        zp0 = rc_mlp(ctx0, "z_prior", prior_act)
        zdraw = zp0[None, :, :32] + rc_noise * zp0[None, :, 32:].clamp(-5, 2).exp()
        vv0 = (rc_base + rc_gaps[(view_gap, gap_act)])[None] + rc_lin(zdraw, "z_head")
        qq0 = rc_norm(vv0 + W0["type_emb.weight"][1], "query_norm")
        wc0 = qq0 + W0["embed.code.weight"][rc_codes] + rc_regs[(age_mode, flag_col)]
        pp0 = (rc_mlp(wc0, "attr_heads.value_bin", head_act)
               + W0["embed_transition.value_bin.weight"][rc_codes * 17 + rc_bins]).softmax(-1)
        md0 = (pp0 * rc_centres).sum(-1).numpy()
        mean0 = md0.mean(0)
        info0 = {
            "cfg": (gap_act, *cfg),
            "rmse": float(np.sqrt(np.mean((mean0 - target_means)**2))),
            "noise_rms_4draw": float(np.sqrt(md0.var(0).mean() / 4)),
            "bias": float((mean0 - target_means).mean()),
        }
        rc_mc_results.append(info0)
print("\nMonte Carlo checks:")
for row in sorted(rc_mc_results, key=lambda x: x["rmse"])[:8]:
    print(row)
```