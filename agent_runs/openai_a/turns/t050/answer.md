The CPU copy uses the same gap scaling for the latent prior and the reading head, although the architecture describes them differently. I’ll test that distinction alongside the plausible register layouts. A large reduction in both mean and likelihood discrepancies would support a reconstruction fix; otherwise I’ll stop guessing and validate edits directly with the forward tool.

```python
torch.set_num_threads(4)
_probe_meta = reference_meta
_probe_sf = state_files[3]
_probe_rng = np.random.default_rng(914)
_probe_ix = np.sort(_probe_rng.choice(len(_probe_meta), 800, replace=False))
_probe_m = _probe_meta.iloc[_probe_ix]
_probe_x = torch.from_numpy(
    _probe_sf["x"][_probe_m.state_row.to_numpy(int)].copy()
)
_probe_dt = torch.tensor(_probe_m.dt_seconds.to_numpy(), dtype=torch.float32)
_probe_h = _probe_x[:, 192:960].reshape(-1, 4, 192)
_probe_flow = _probe_h * torch.exp(
    -_probe_dt[:, None, None] * torch.exp(W["core.flow.a"])[None, None, :]
)
_probe_context = _probe_x[:, :192] + ln(_probe_flow.mean(1), "core.out_norm")
_probe_t = torch.log1p(_probe_dt)[:, None]
_probe_prior = mlp(_probe_context + mlp(_probe_t / 10, "gap"), "z_prior")
_probe_codes = torch.tensor(_probe_m.code.to_numpy(), dtype=torch.long)
_probe_bins = torch.tensor(_probe_m.last_bin.to_numpy(), dtype=torch.long)
_probe_has = torch.tensor(_probe_m.has_last.to_numpy(), dtype=torch.float32)
_probe_z = torch.tensor(_probe_m.last_z.to_numpy(), dtype=torch.float32)
_probe_age = torch.log1p(torch.tensor(
    _probe_m.last_age_seconds.to_numpy(), dtype=torch.float32
)) / 10
_probe_fixed = (
    W["embed.code.weight"][_probe_codes]
    + W["embed.bins.value.weight"][_probe_bins]
    + W["embed.bins.rate.weight"][0]
)
_probe_trans = W["embed_transition.value_bin.weight"][_probe_codes * 17 + _probe_bins]
_probe_centres = head_centres[_probe_ix]
_probe_obs = head_obsbins[_probe_ix].reshape(-1, 1)
_probe_gpu_mu = np.asarray(as_numpy(head_gpu_mean))[_probe_ix]
_probe_gpu_nll = np.asarray(as_numpy(head_gpu_nll))[_probe_ix]
_probe_results = []

print("Absent dose-bin norm:", float(W["embed.bins.dose.weight"][0].norm()))
for _gs in (0.1, 1.0):
    for _zs in (0.0, 0.5, 1.0):
        _probe_q = ln(
            _probe_context + mlp(_probe_t * _gs, "gap")
            + lin(_probe_prior[:, :32] * _zs, "z_head")
            + W["type_emb.weight"][1], "query_norm"
        )
        for _vc in (2, 4, 6):
            for _flags in ((3,), (5,), (7,), (3, 5, 7)):
                _reg = torch.zeros(len(_probe_m), 8)
                _reg[:, 0] = _probe_has
                _reg[:, 1] = _probe_age
                _reg[:, _vc] = _probe_z
                for _fc in _flags:
                    _reg[:, _fc] = _probe_has
                for _status in (0, 1):
                    _add = _probe_fixed + lin(_reg, "last_numeric") + W["last_status.weight"][_status]
                    _hidden = F.gelu(lin(_probe_q + _add, "attr_heads.value_bin.0"))
                    _logits = lin(_hidden, "attr_heads.value_bin.2") + _probe_trans
                    _prob = _logits.softmax(-1)
                    _mu = (_prob * _probe_centres).sum(-1).numpy()
                    _nl = -_prob.gather(1, _probe_obs).clamp_min(1e-20).log().squeeze(1).numpy()
                    _mean_scale = np.array([
                        norms[(e, "value")]["iqr"] for e in _probe_m.event
                    ])
                    _score = np.mean(((_mu - _probe_gpu_mu) / _mean_scale) ** 2) \
                             + 0.02 * np.mean((_nl - _probe_gpu_nll) ** 2)
                    _probe_results.append({
                        "gap_scale": _gs, "z_scale": _zs, "value_col": _vc,
                        "flags": str(_flags), "status": _status,
                        "score": float(_score),
                        "rmse": float(np.sqrt(np.mean((_mu - _probe_gpu_mu) ** 2))),
                        "nll": float(_nl.mean()),
                        "nll_difference_rmse": float(np.sqrt(np.mean((_nl - _probe_gpu_nll) ** 2)))
                    })
_probe_results = pd.DataFrame(_probe_results).sort_values("score")
print(_probe_results.head(12).to_string(index=False))
with Path("/work/notes.md").open("a") as _f:
    _f.write("\n## Restart diagnostics; Python 44, forward 4\n"
             "- No edits. Testing separate prior/head gap scaling and register layouts.\n"
             "- Earlier insulin identification supports beta about -1.93 and persistence .832;\n"
             "  norepinephrine remains less securely identified.\n")
```
