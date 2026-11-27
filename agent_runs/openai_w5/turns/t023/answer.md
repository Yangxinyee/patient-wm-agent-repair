The informative insulin strata agree on a negative next-hour effect, roughly −1.8 to −2.0 per rate unit. Norepinephrine’s active stratum suggests about +3.2; the near-zero strata have too little dose variation to be informative. The linear dynamic model implies tuning-set shifts near −11.9, +5.95, +4.50, and −2.25, but these remain assumption-dependent estimates.

Bin-edge changes did not fix the head reconstruction. I will now test the eight-column register layouts systematically. Agreement on both means and NLL in a separate sample would validate a layout.

```python
import time
_t0 = time.perf_counter()

def head_features_layout(cache, layout, ageunit="seconds", subset=None,
                         nsamples=16, seed=913):
    ii = slice(None) if subset is None else subset
    base = torch.as_tensor(cache["base"])[ii]
    dt = torch.as_tensor(cache["dt"])[ii]
    gapv = mlp0(torch.log1p(dt) / 10, "gap", "gelu")
    zp = mlp0(base + gapv, "z_prior", "gelu")
    gen = torch.Generator().manual_seed(seed)
    eps = torch.randn((nsamples, len(base), 32), generator=gen)
    z = zp[None, :, :32] + eps * zp[:, 32:].clamp(-6, 2).exp()[None]
    q = norm0(base[None] + gapv[None] + lin0(z, "z_head")
              + W0["type_emb.weight"][1], "query_norm")
    has = torch.as_tensor(cache["haslast"])[ii]
    age = torch.as_tensor(cache["lastage"])[ii]
    if ageunit != "seconds":
        age = torch.log1p(torch.expm1(age * 10) /
                         (60 if ageunit == "minutes" else 3600)) / 10
    nums = torch.zeros((len(base), 8))
    vc, ac, flags = layout
    nums[:, vc] = torch.as_tensor(cache["lastval"])[ii]
    nums[:, ac] = age
    for fc in flags:
        nums[:, fc] = has
    codes = torch.as_tensor(cache["codes"], dtype=torch.long)[ii]
    bins = torch.as_tensor(cache["lastbin"], dtype=torch.long)[ii]
    reg = (W0["embed.code.weight"][codes] + lin0(nums, "last_numeric")
           + W0["embed.bins.value.weight"][bins]
           + W0["embed.bins.rate.weight"][0])
    wc = q + reg[None]
    hidden = act0(lin0(wc, "attr_heads.value_bin.0"), "gelu")
    logits = (lin0(hidden, "attr_heads.value_bin.2")
              + W0["embed_transition.value_bin.weight"][codes * 17 + bins][None])
    probs = logits.softmax(-1).mean(0)
    centers = torch.as_tensor(cache["centers"])[ii]
    means = (probs * centers).sum(-1)
    targets = torch.as_tensor(cache["targetbin"], dtype=torch.long)[ii]
    nll = -probs.gather(1, targets[:, None]).squeeze(1).clamp_min(1e-12).log()
    return dict(hidden=hidden, wc=wc, logits=logits, probs=probs,
                means=means, nll=nll)

_probe = np.sort(np.random.default_rng(61).choice(len(cacheA["df"]), 256, replace=False))
_baseline = head_features_layout(cacheA, (2, 1, (0, 3)), subset=_probe, nsamples=8)
_ref_mean, _ref_nll = reconstruct_head(cacheA, _probe, nsamples=8, **head_spec)
print("algebra check, max mean discrepancy:",
      float((_baseline["means"] - _ref_mean).abs().max()))
assert (_baseline["means"] - _ref_mean).abs().max() < 1e-3

_base = torch.as_tensor(cacheA["base"])[_probe]
_dt = torch.as_tensor(cacheA["dt"])[_probe]
_gap = mlp0(torch.log1p(_dt) / 10, "gap", "gelu")
_zp = mlp0(_base + _gap, "z_prior", "gelu")
_gen = torch.Generator().manual_seed(913)
_eps = torch.randn((8, len(_probe), 32), generator=_gen)
_z = _zp[None, :, :32] + _eps * _zp[:, 32:].clamp(-6, 2).exp()[None]
_q = norm0(_base[None] + _gap[None] + lin0(_z, "z_head")
           + W0["type_emb.weight"][1], "query_norm")
_codes = torch.as_tensor(cacheA["codes"], dtype=torch.long)[_probe]
_has = torch.as_tensor(cacheA["haslast"])[_probe, None]
_lv = torch.as_tensor(cacheA["lastval"])[_probe, None]
_age0 = torch.as_tensor(cacheA["lastage"])[_probe]
_ages = {
    "seconds": _age0[:, None],
    "minutes": (torch.log1p(torch.expm1(_age0 * 10) / 60) / 10)[:, None],
    "hours": (torch.log1p(torch.expm1(_age0 * 10) / 3600) / 10)[:, None]
}
_dirs = W0["attr_heads.value_bin.0.weight"] @ W0["last_numeric.weight"]
_centers = torch.as_tensor(cacheA["centers"])[_probe]
_gm = torch.tensor(np.asarray(cacheA["gpu_mean"])[_probe], dtype=torch.float32)
_gn = torch.tensor(np.asarray(cacheA["gpu_nll"])[_probe], dtype=torch.float32)
_sc = torch.tensor([norm_meta[(e, "value")]["iqr"]
                    for e in cacheA["df"].event.iloc[_probe]])
layout_grid = []

for _side, _cache in (("left", cacheA), ("right", cacheA_right)):
    _bins = torch.as_tensor(_cache["lastbin"], dtype=torch.long)[_probe]
    _targets = torch.as_tensor(_cache["targetbin"], dtype=torch.long)[_probe]
    _reg0 = (W0["embed.code.weight"][_codes]
             + W0["embed.bins.value.weight"][_bins]
             + W0["embed.bins.rate.weight"][0] + W0["last_numeric.bias"])
    _static = lin0(_q + _reg0[None], "attr_heads.value_bin.0")
    _trans = W0["embed_transition.value_bin.weight"][_codes * 17 + _bins]
    for _vc, _ac in itertools.permutations(range(8), 2):
        _available = [j for j in range(8) if j not in (_vc, _ac)]
        _flagsets = [()] + [(j,) for j in _available] + list(
            itertools.combinations(_available, 2))
        for _flags, (_unit, _age) in itertools.product(_flagsets, _ages.items()):
            _const = (_dirs[:, list(_flags)].sum(1) if _flags
                      else torch.zeros(192))
            _aff = (_static + (_lv * _dirs[:, _vc] +
                              _age * _dirs[:, _ac] + _has * _const)[None])
            _logits = lin0(act0(_aff, "gelu"), "attr_heads.value_bin.2") + _trans[None]
            _p = _logits.softmax(-1).mean(0)
            _m = (_p * _centers).sum(-1)
            _n = -_p.gather(1, _targets[:, None]).squeeze(1).clamp_min(1e-12).log()
            _score = float(((_m - _gm) / _sc).square().mean())
            layout_grid.append((_score, _side, _vc, _ac, _flags, _unit,
                                float((_n - _gn).abs().mean())))
layout_grid.sort(key=lambda x: x[0])
print("\n=== COARSE REGISTER SEARCH ===")
for _row in layout_grid[:8]:
    print(_row)

_refine = np.sort(np.random.default_rng(62).choice(len(cacheA["df"]), 768, replace=False))
_gm2 = torch.tensor(np.asarray(cacheA["gpu_mean"])[_refine], dtype=torch.float32)
_gn2 = torch.tensor(np.asarray(cacheA["gpu_nll"])[_refine], dtype=torch.float32)
_sc2 = torch.tensor([norm_meta[(e, "value")]["iqr"]
                     for e in cacheA["df"].event.iloc[_refine]])
layout_refined = []
for _row in layout_grid[:24]:
    _, _side, _vc, _ac, _flags, _unit, _ = _row
    _cache = cacheA if _side == "left" else cacheA_right
    _res = head_features_layout(_cache, (_vc, _ac, _flags), _unit,
                                subset=_refine, nsamples=64)
    _score = float(((_res["means"] - _gm2) / _sc2).square().mean())
    layout_refined.append((_score, _side, _vc, _ac, _flags, _unit,
                           float((_res["nll"] - _gn2).abs().mean())))
layout_refined.sort(key=lambda x: x[0])
print("\n=== INDEPENDENT-SAMPLE REFINEMENT ===")
for _row in layout_refined[:8]:
    print(_row)

_best = layout_refined[0]
head_layout_spec = dict(binside=_best[1], layout=(_best[2], _best[3], _best[4]),
                        ageunit=_best[5])
print("selected:", head_layout_spec)
_res = head_features_layout(
    cacheA if _best[1] == "left" else cacheA_right,
    head_layout_spec["layout"], head_layout_spec["ageunit"], nsamples=32)
for _event in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff"):
    _mask = torch.tensor((cacheA["df"].event == _event).to_numpy())
    _err = (_res["means"] - torch.tensor(np.asarray(cacheA["gpu_mean"])))[_mask]
    print(_event, "bias", float(_err.mean()),
          "RMSE", float(_err.square().mean().sqrt()))
print("NLL MAE:", float((_res["nll"] -
                       torch.tensor(np.asarray(cacheA["gpu_nll"]))).abs().mean()))
print("elapsed seconds:", round(time.perf_counter() - _t0, 1))

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Register search / run 19\n"
             f"- Selected reconstruction convention: {head_layout_spec}.\n"
             "- Insulin slopes agree across informative strata; near-zero-dose "
             "strata have inadequate innovation variance and do not identify effects.\n"
             "- Provisional AR(1) tuning-A targets: insulin double -11.893, half "
             "+5.946; NE double +4.497, half -2.249. These are estimated, not truth.\n"
             "- No weight edits.\n")
```
