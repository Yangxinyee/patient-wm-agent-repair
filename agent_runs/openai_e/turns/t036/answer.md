The approximate posterior features recover some insulin innovation but essentially none for norepinephrine. That is not enough for a reliable final-layer edit. The errors cluster around infusion endings, so I’ll test how END atoms are routed, plus a small measurement-kind check. Numerical agreement would confirm the convention; continued mismatch would rule it out.

```python
def _post33(indices, e, u):
    c = recon_ctx(indices, act="gelu", gap_style="log")
    h = torch.nn.functional.gelu(torch.nn.functional.linear(
        torch.cat([c, e, u], dim=1),
        W0["z_post.0.weight"], W0["z_post.0.bias"]))
    return torch.nn.functional.linear(
        h, W0["z_post.2.weight"], W0["z_post.2.bias"])[:, :32]

# Test END routing on drug bundles, with either absent or retained numbers.
_ix33 = np.asarray(action_rows, dtype=int)
_startobs33, _startact33, _endabs33, _endfull33 = [], [], [], []
for _j33 in _ix33:
    _key33 = (int(S0["stay_id"][_j33]), int(round(S0["hour"][_j33]*60)))
    _st33 = record_starts.get(_key33, [])
    _en33 = record_ends.get(_key33, [])
    _startobs33.append([recon_record_atom(r) for r in _st33
                       if r.event not in ("insulin", "norepinephrine")])
    _startact33.append([recon_record_atom(r) for r in _st33
                       if r.event in ("insulin", "norepinephrine")])
    _endabs33.append([recon_record_atom(
        r, kind=2, end_numeric="absent", end_units="code") for r in _en33])
    _endfull33.append([recon_record_atom(r, kind=0)
                      - W0["embed.kind.weight"][0]
                      + W0["embed.kind.weight"][2] for r in _en33])

_route_results33 = []
for _numlabel33, _endatoms33 in (("absent", _endabs33), ("retained", _endfull33)):
    for _route33 in ("action", "observation", "ignored"):
        _elist33 = [o + (en if _route33 == "observation" else [])
                    for o, en in zip(_startobs33, _endatoms33)]
        _ulist33 = [a + (en if _route33 == "action" else [])
                    for a, en in zip(_startact33, _endatoms33)]
        _ee33 = recon_sets(_elist33, prefix="encode", empty_zero=False)
        _uu33 = recon_sets(_ulist33, prefix="encode_action", empty_zero=True)
        _pred33 = _post33(_ix33, _ee33, _uu33).numpy()
        _er33 = np.sqrt(np.mean((_pred33-S0["x"][_ix33, 960:])**2, axis=1))
        _row33 = (_numlabel33, _route33,
                  round(float(np.median(_er33)), 6),
                  round(float(np.sqrt(np.mean(_er33**2))), 6),
                  int((_er33 < .002).sum()))
        _route_results33.append(_row33)
print("=== END ROUTING: numbers, route, median, RMS, verified ===")
print(_route_results33)

# Singleton measurement bundles: check kind and key-quality conventions.
_measure_results33 = []
for _event33 in ("glucose_finger", "map_arterial", "map_cuff"):
    _js33 = np.asarray(_groups31.get((_event33, "none"), [])[:24], dtype=int)
    if len(_js33) == 0:
        continue
    _atoms33 = []
    for _j33 in _js33:
        _key33 = (int(S0["stay_id"][_j33]), int(round(S0["hour"][_j33]*60)))
        _atoms33.append(recon_record_atom(record_starts[_key33][0]))
    _base33 = torch.stack(_atoms33)
    _uempty33 = torch.zeros((len(_js33), 192))
    _tests33 = []
    for _kind33 in (0, 1, 2):
        for _kq33 in range(SHAPES["embed.kq.weight"][0]):
            _aa33 = (_base33 - W0["embed.kind.weight"][0]
                     + W0["embed.kind.weight"][_kind33]
                     - W0["embed.kq.weight"][0]
                     + W0["embed.kq.weight"][_kq33])
            _ee33 = recon_single_set(_aa33, prefix="encode")
            _pr33 = _post33(_js33, _ee33, _uempty33).numpy()
            _rm33 = float(np.sqrt(np.mean(
                (_pr33-S0["x"][_js33, 960:])**2)))
            _tests33.append((_kind33, _kq33, _rm33))
    _measure_results33.append(
        (_event33, sorted(_tests33, key=lambda x: x[2])[:3]))
print("=== MEASUREMENT KIND / QUALITY: best three ===")
print(_measure_results33)

with open("/work/notes.md", "a") as _f33:
    _f33.write(
        "\n## Run 33: targeted encoding convention tests\n"
        f"END routing results: {_route_results33}.\n"
        f"Measurement kind/quality results: {_measure_results33}.\n"
        "Unchanged posterior basis recovered some insulin innovation but no "
        "reliable NE innovation. No edits. Python runs: 33; forward passes: 3.\n"
    )
```