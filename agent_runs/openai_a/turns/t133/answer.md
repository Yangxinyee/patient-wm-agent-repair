There are 48 clean two-reading bundles available for validation. I’ll first reproduce the previously exact insulin posterior, then test measurement embedding choices under that same context formula. This separates a time/context error from an atom-encoding error; a near-zero measurement error is the acceptance criterion.

```python
import itertools, time
assert not torch.is_grad_enabled()
torch.set_num_threads(8)
r118_start = time.time()

# First revalidate the latent context on the previously identified insulin bundles.
r118_drug_atoms = rm_insulin_atoms(3, 2, "left", True, "gelu")
r118_drug_u = rm_set(r118_drug_atoms, "encode_action")
r118_drug_e = r96_empty_world(0)
r118_context_trials = []
for time_style in ("log", "log1p"):
    for scale in (1.0, 10.0):
        for flowed in (True, False):
            d = _r96_dt.clamp_min(1e-6)
            hp = _r96_prev[:, 192:960].reshape(-1, 4, 192)
            hf = hp * torch.exp(-d[:, :, None] * torch.exp(rm_W["core.flow.a"])[None, None])
            ht = hf if flowed else hp
            tv = (torch.log(d) if time_style == "log" else torch.log1p(d)) / scale
            cx = _r96_prev[:, :192] + rm_ln(ht.mean(1), "core.out_norm") + rm_mlp(tv, "gap")
            zz = rm_mlp(torch.cat([cx, r118_drug_e, r118_drug_u], 1), "z_post")[:, :32]
            err = float(torch.mean((zz - _r96_target)**2).sqrt())
            r118_context_trials.append((err, time_style, scale, flowed))
r118_context_trials.sort()
print("INSULIN CONTEXT CHECK:", r118_context_trials[:4])
_, r118_time_style, r118_time_scale, r118_flowed = r118_context_trials[0]

# Reconstruct both observed atoms from the actual two-reading bundles.
r118_B = len(r117_ids)
r118_prev = torch.tensor(rm_states["x"][r117_ids-1])
r118_dt = torch.tensor(
    (rm_states["hour"][r117_ids] - rm_states["hour"][r117_ids-1])[:, None] * 3600,
    dtype=torch.float32
)
r118_hp = r118_prev[:, 192:960].reshape(-1, 4, 192)
r118_hf = r118_hp * torch.exp(-r118_dt[:, :, None] * torch.exp(rm_W["core.flow.a"])[None, None])
r118_ht = r118_hf if r118_flowed else r118_hp
r118_tv = (torch.log(r118_dt) if r118_time_style == "log" else torch.log1p(r118_dt)) / r118_time_scale
r118_ctx = r118_prev[:, :192] + rm_ln(r118_ht.mean(1), "core.out_norm") + rm_mlp(r118_tv, "gap")
r118_target = torch.tensor(rm_states["x"][r117_ids, 960:992])
r118_values = np.array([
    [float(g.loc[g.event == ev, "value"].iloc[0])
     for ev in ("glucose_finger", "map_arterial")]
    for g in r117_groups_used
], dtype=np.float32)
r118_num = torch.zeros(r118_B, 2, 6)
r118_bins = torch.empty(r118_B, 2, dtype=torch.long)
for j, ev in enumerate(("glucose_finger", "map_arterial")):
    ns = rm_normspec[(ev, "value")]
    r118_num[:, j, 0] = torch.tensor(np.clip(
        (r118_values[:, j] - ns["median"]) / ns["iqr"], -10, 10))
    r118_num[:, j, 1] = 1
    r118_bins[:, j] = torch.tensor(
        1 + np.searchsorted(ns["fine_bin_edges"], r118_values[:, j], side="left"))
r118_base = (
    rm_W["embed.code.weight"][torch.tensor([6, 7])][None]
    + rm_W["embed.kind.weight"][0]
    + rm_lin(r118_num, "embed.numeric")
    + rm_W["embed.bins.value.weight"][r118_bins]
)

r118_unit_table = rm_W["embed.categorical.unit.weight"]
r118_kq_table = rm_W["embed.kq.weight"]
print("ROW-ZERO NORMS:", {
    n: float(rm_W[n][0].norm()) for n in (
        "embed.categorical.unit.weight", "embed.kq.weight",
        "embed.bins.dose.weight", "embed.bins.rate.weight", "embed.bins.value.weight")
})
r118_u_options = {
    "zero": torch.zeros(r118_B, 192),
    "empty_encoder": empty_set_vector("encode_action", r118_B)
}
r118_specs = list(itertools.product(range(5), range(-1, 4), range(-1, 6), range(-1, 6)))
r118_scores = []
for start in range(0, len(r118_specs), 32):
    specs = r118_specs[start:start+32]
    atoms = r118_base[None].repeat(len(specs), 1, 1, 1)
    for j, (ek, kq, gu, mu) in enumerate(specs):
        atoms[j] += rm_W["embed.ekind.weight"][ek]
        if kq >= 0:
            atoms[j] += r118_kq_table[kq]
        if gu >= 0:
            atoms[j, :, 0] += r118_unit_table[gu]
        if mu >= 0:
            atoms[j, :, 1] += r118_unit_table[mu]
    es = rm_set(atoms.reshape(-1, 2, 192), "encode")
    for uname, uv in r118_u_options.items():
        zz = rm_mlp(torch.cat([
            r118_ctx.repeat(len(specs), 1), es, uv.repeat(len(specs), 1)
        ], 1), "z_post")[:, :32].reshape(len(specs), r118_B, 32)
        errs = ((zz - r118_target[None])**2).mean((1, 2)).sqrt().tolist()
        r118_scores.extend((err, spec, uname) for err, spec in zip(errs, specs))
r118_scores.sort()
print("BEST MEASUREMENT CHECKS:", r118_scores[:10])
print("SECONDS:", round(time.time() - r118_start, 1))
```
