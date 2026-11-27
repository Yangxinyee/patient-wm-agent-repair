The first posterior reconstruction still misses the exposed means by about 0.13 RMS, so it is not yet reliable enough for an edit. I’ll test activation functions, empty observation-bundle encoding, and the posterior’s relation to the prior. Agreement below roughly 0.0001 would confirm the reconstruction; persistent error would refute these conventions.

```python
repair_acts = {"gelu": F.gelu, "silu": F.silu}
def repair_mlp_act(x, name, activation):
    return repair_linear(activation(repair_linear(x, name + ".0")), name + ".2")

repair_empty_e = {"zero": torch.zeros(192)}
for _act_name, _act in repair_acts.items():
    _pool = repair_linear(torch.zeros(1), "encode.size")
    repair_empty_e["empty_" + _act_name] = repair_norm(
        repair_mlp_act(_pool, "encode.mlp2", _act), "encode.norm"
    )

repair_posterior_search = []
repair_posterior_best = {}
_start = time.time()

for _drug in ("insulin", "norepinephrine"):
    _cases = [(j, r) for j, r in repair_single_rows if r["event"] == _drug][:24]
    if not _cases:
        continue
    _js = np.array([j for j, _ in _cases])
    _previous = torch.from_numpy(state_x[_js-1])
    _dt = torch.tensor((state_hours[_js] - state_hours[_js-1]) * 3600,
                       dtype=torch.float32)
    _hflow = _previous[:, 192:960].reshape(-1, 4, 192) * torch.exp(
        -_dt[:, None, None] * torch.exp(repair_weights["core.flow.a"])[None, None, :]
    )
    _base_ctx = _previous[:, :192] + repair_norm(_hflow.mean(1), "core.out_norm")
    _target = torch.from_numpy(state_x[_js, 960:992])
    _atoms0 = torch.stack([repair_atom_base(r) for _, r in _cases])
    _contexts = {
        name: _base_ctx + repair_mlp_act(
            (torch.log(_dt) / 10)[:, None], "gap", act
        ) for name, act in repair_acts.items()
    }
    print(_drug, "exposed latent mean/std:",
          round(float(_target.mean()), 4), round(float(_target.std()), 4))
    for _gap_name, _ctx in _contexts.items():
        for _prior_name, _prior_act in repair_acts.items():
            _prior = repair_mlp_act(_ctx, "z_prior", _prior_act)[:, :32]
            print(" prior-only", _gap_name, _prior_name,
                  round(float(((_prior - _target)**2).mean().sqrt()), 6))

    _best_records = {}
    for _start in range(0, len(repair_cat_grid), 64):
        _configs = repair_cat_grid[_start:_start+64]
        _adds = []
        for unit, rate_unit, action_cat, route, status, kq in _configs:
            _a = sum((
                repair_weights["embed.categorical.unit.weight"][unit],
                repair_weights["embed.categorical.rate_unit.weight"][rate_unit],
                repair_weights["embed.categorical.action.weight"][action_cat],
                repair_weights["embed.categorical.route.weight"][route],
                repair_weights["embed.categorical.status.weight"][status],
            ))
            if kq >= 0:
                _a = _a + repair_weights["embed.kq.weight"][kq]
            _adds.append(_a)
        _atoms = _atoms0[None] + torch.stack(_adds)[:, None]
        for _enc_name, _enc_act in repair_acts.items():
            _first = repair_mlp_act(_atoms, "encode_action.mlp1", _enc_act)
            _pooled = _first + repair_linear(
                torch.full((*_first.shape[:-1], 1), np.log(2)), "encode_action.size"
            )
            _actions = repair_norm(
                repair_mlp_act(_pooled, "encode_action.mlp2", _enc_act),
                "encode_action.norm"
            )
            for _gap_name, _ctx in _contexts.items():
                _cx = _ctx[None].expand(len(_configs), -1, -1)
                for _empty_name, _empty in repair_empty_e.items():
                    _ev = _empty.expand_as(_actions)
                    _input = torch.cat([_cx, _ev, _actions], -1)
                    for _post_name, _post_act in repair_acts.items():
                        _post = repair_mlp_act(_input, "z_post", _post_act)
                        for _layout, _mu in (
                            ("chunk", _post[..., :32]),
                            ("interleaved", _post[..., ::2]),
                        ):
                            _predictions = [("none", _mu)]
                            for _prior_name, _prior_act in repair_acts.items():
                                _prior = repair_mlp_act(_ctx, "z_prior", _prior_act)[:, :32]
                                _predictions.append((_prior_name, _mu + _prior[None]))
                            for _prior_name, _prediction in _predictions:
                                _errors = ((_prediction - _target[None])**2).mean((1, 2)).sqrt()
                                _k = int(_errors.argmin())
                                _key = (_enc_name, _gap_name, _empty_name,
                                        _post_name, _layout, _prior_name)
                                _rms = float(_errors[_k])
                                if _key not in _best_records or _rms < _best_records[_key][0]:
                                    _best_records[_key] = (_rms, _configs[_k])
    _ranked = sorted(
        [(v[0], k, v[1]) for k, v in _best_records.items()],
        key=lambda x: x[0]
    )
    repair_posterior_best[_drug] = _ranked[0]
    print("Best 6 conventions:", _ranked[:6])
    repair_posterior_search.extend(
        [(_drug, rms, convention, config) for rms, convention, config in _ranked]
    )

print("Elapsed seconds:", round(time.time() - _start, 2))
with open("/work/notes.md", "a") as _f:
    _f.write(
        "\n## Python 36; forward 6\n"
        "- First posterior reconstruction RMS ~0.13, not exact; no edit made.\n"
        "- Tested GELU/SiLU, empty-observation encoding, latent layouts, and prior addition.\n"
        f"- Best reconstruction conventions: {repair_posterior_best!r}\n"
    )
```