The reconstruction appears to omit the last-status embedding and assumes one gap-scaling convention. Those omissions could explain why an implausible register layout scored best. I will test the documented register layout with alternative head conventions, without editing weights. A substantial reduction in error would support this explanation.

```python
import itertools
import torch.nn.functional as F
torch.set_num_threads(4)

print("event kinds:", VOCAB["event_kinds"])
print("atom kinds:", VOCAB["atom_kinds"])
print("numeric note:", VOCAB.get("numeric_note"))
print("panel columns:", list(panel.columns))
print("X columns:", list(X.columns))
print("cache shapes:", {
    k: getattr(v, "shape", None) for k, v in cacheA.items()
})
for _name in ("lin0", "norm0", "mlp0", "head0"):
    print(_name, "constants:", globals()[_name].__code__.co_consts)
print("last-status norms:",
      torch.linalg.vector_norm(W0["last_status.weight"], dim=1).tolist())

def reconstruct_head(cache, subset=None, kind="gelu", typeidx=1,
                     prior_gap=10.0, view_gap=10.0, status=-1,
                     ageunit="seconds", flagcol=3, nsamples=0,
                     sd_power=1.0, seed=913, keep_hidden=False):
    ii = slice(None) if subset is None else subset
    base = torch.as_tensor(cache["base"])[ii]
    dt = torch.as_tensor(cache["dt"])[ii]
    g = torch.log1p(dt)
    ctx = base + mlp0(g / prior_gap, "gap", kind)
    zp = mlp0(ctx, "z_prior", kind)
    mu = zp[..., :32]
    if nsamples:
        gen = torch.Generator().manual_seed(seed)
        eps = torch.randn((nsamples, len(base), 32), generator=gen)
        z = mu[None] + eps * torch.exp(
            sd_power * zp[..., 32:].clamp(-6, 2))[None]
    else:
        z = mu[None]
    view = base[None] + mlp0(g / view_gap, "gap", kind)[None] + lin0(z, "z_head")
    q = norm0(view + W0["type_emb.weight"][typeidx], "query_norm")
    has = torch.as_tensor(cache["haslast"])[ii]
    nums = torch.zeros((len(base), 8))
    nums[:, 0] = has
    age = torch.as_tensor(cache["lastage"])[ii]
    if ageunit != "seconds":
        age = torch.log1p(torch.expm1(age * 10) /
                         (3600 if ageunit == "hours" else 60)) / 10
    nums[:, 1] = age
    nums[:, 2] = torch.as_tensor(cache["lastval"])[ii]
    if flagcol is not None:
        nums[:, flagcol] = has
    codes = torch.as_tensor(cache["codes"], dtype=torch.long)[ii]
    lastbin = torch.as_tensor(cache["lastbin"], dtype=torch.long)[ii]
    reg = (W0["embed.code.weight"][codes] + lin0(nums, "last_numeric")
           + W0["embed.bins.value.weight"][lastbin]
           + W0["embed.bins.rate.weight"][0])
    if status >= 0:
        reg = reg + W0["last_status.weight"][status]
    wc = q + reg[None]
    hidden = act0(lin0(wc, "attr_heads.value_bin.0"), kind)
    logits = (lin0(hidden, "attr_heads.value_bin.2")
              + W0["embed_transition.value_bin.weight"][codes * 17 + lastbin][None])
    probs = logits.softmax(-1).mean(0)
    centers = torch.as_tensor(cache["centers"])
    if centers.ndim > 1:
        centers = centers[ii]
    means = (probs * centers).sum(-1)
    target = torch.as_tensor(cache["targetbin"], dtype=torch.long)[ii]
    nll = -torch.log(probs.gather(1, target[:, None]).squeeze(1).clamp_min(1e-12))
    if keep_hidden:
        return means, nll, probs, hidden, wc, logits
    return means, nll

_probe = np.sort(np.random.default_rng(41).choice(len(cacheA["df"]), 192, replace=False))
_gm = torch.as_tensor(cacheA["gpu_mean"], dtype=torch.float32)[_probe]
_gn = torch.as_tensor(cacheA["gpu_nll"], dtype=torch.float32)[_probe]
_sc = torch.tensor([
    norm_meta[(e, "value")]["iqr"] for e in cacheA["df"].event.iloc[_probe]
], dtype=torch.float32)

convention_search = []
for _kind, _pg, _vg, _type, _status in itertools.product(
        ("gelu", "relu", "silu"), (10.0, 1.0), (10.0, 1.0),
        range(15), (-1, 0, 1)):
    _pm, _pn = reconstruct_head(
        cacheA, _probe, _kind, _type, _pg, _vg, _status)
    _me = ((_pm - _gm) / _sc).square().mean().item()
    _ne = (_pn - _gn).square().mean().item()
    convention_search.append((_me + 0.03 * _ne, _me, _ne,
                              _kind, _pg, _vg, _type, _status))
convention_search.sort()
print("\n=== BEST DOCUMENTED-LAYOUT CONVENTIONS ===")
for _row in convention_search[:10]:
    print(_row)

_top = convention_search[0]
head_spec = dict(kind=_top[3], prior_gap=_top[4], view_gap=_top[5],
                 typeidx=_top[6], status=_top[7])
print("\n=== MONTE CARLO CHECK ===", head_spec)
for _samples in (0, 4, 32):
    _pm, _pn = reconstruct_head(cacheA, nsamples=_samples, **head_spec)
    _gm_all = torch.as_tensor(cacheA["gpu_mean"])
    _gn_all = torch.as_tensor(cacheA["gpu_nll"])
    print("samples", _samples,
          "NLL MAE", (_pn - _gn_all).abs().mean().item(),
          "NLL bias", (_pn - _gn_all).mean().item())
    for _event in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff"):
        _mask = torch.as_tensor((cacheA["df"].event == _event).to_numpy())
        _err = (_pm - _gm_all)[_mask]
        print(_event, "bias", round(_err.mean().item(), 4),
              "RMSE", round(_err.square().mean().sqrt().item(), 4))

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Reconstruction conventions / run 16\n"
             "- Tested schema-valid last-numeric features, last-status embeddings, "
             "type embeddings, activations, and prior/head gap scaling.\n"
             f"- Best deterministic convention: {head_spec}.\n"
             "- No tensor edits; sampling and register conventions still require validation.\n")
```
