Routing END atoms through the observation encoder reduced the drug-bundle error dramatically. That supports the routing hypothesis; the small remaining error may be an attribute convention. I’ll resolve it and test whether the action encoder itself retains doses. I’ll also check whether measurement inputs use recorded values or bin centres, since their mismatch remains unexplained.

```python
# Refine the END convention under the now-supported observation routing.
_endtests34 = []
_best_end34 = None
_best_score34 = np.inf
for _units34 in ("code", "none"):
    for _ekind34 in (2, 0):
        _ends34 = []
        for _j34 in _ix33:
            _key34 = (int(S0["stay_id"][_j34]), int(round(S0["hour"][_j34]*60)))
            _lst34 = []
            for _r34 in record_ends.get(_key34, []):
                _a34 = recon_record_atom(
                    _r34, kind=2, end_numeric="absent", end_units=_units34)
                _a34 = (_a34 - W0["embed.ekind.weight"][RECON_EKINDS[_r34.event]]
                        + W0["embed.ekind.weight"][_ekind34])
                _lst34.append(_a34)
            _ends34.append(_lst34)
        _e34 = recon_sets([o+en for o, en in zip(_startobs33, _ends34)],
                          prefix="encode", empty_zero=False)
        _u34 = recon_sets(_startact33, prefix="encode_action", empty_zero=True)
        _z34 = _post33(_ix33, _e34, _u34).numpy()
        _err34 = np.sqrt(np.mean((_z34-S0["x"][_ix33, 960:])**2, axis=1))
        _score34 = float(np.sqrt(np.mean(_err34**2)))
        _endtests34.append((_units34, _ekind34, round(_score34, 7),
                           int((_err34 < .0001).sum())))
        if _score34 < _best_score34:
            _best_score34 = _score34
            _best_end34 = (_units34, _ekind34)
            _act_e34, _act_u34, _act_err34 = _e34, _u34, _err34
print("=== END refinement: units, ekind, RMS, numerical matches ===")
print(_endtests34)

_act_ctx34 = recon_ctx(_ix33, act="gelu", gap_style="log")
_act_input34 = torch.cat([_act_ctx34, _act_e34, _act_u34], dim=1)
_act_hidden34 = torch.nn.functional.gelu(torch.nn.functional.linear(
    _act_input34, W0["z_post.0.weight"], W0["z_post.0.bias"])).numpy()

# Can the incoming action vector recover doses, before the posterior bottleneck?
_U34 = _act_u34.numpy().astype(np.float64)
_tr34 = S0["stay_id"][_ix33] % 5 != 0
_te34 = ~_tr34
_umu34 = _U34[_tr34].mean(axis=0)
_usd34 = np.maximum(_U34[_tr34].std(axis=0), .05)
_UX34 = (_U34-_umu34)/_usd34
_udose34 = Ridge(alpha=1.0).fit(_UX34[_tr34], action_raw[_tr34])
_uinnov34 = Ridge(alpha=1.0).fit(_UX34[_tr34], action_targets[_tr34])
print("=== ACTION-ENCODER held-out R2 ===")
print("raw dose:", np.round(r2_score(action_raw[_te34],
      _udose34.predict(_UX34[_te34]), multioutput="raw_values"), 4))
print("innovation:", np.round(r2_score(action_targets[_te34],
      _uinnov34.predict(_UX34[_te34]), multioutput="raw_values"), 4))

# Test numeric value conventions on singleton measurement bundles.
_value_tests34 = []
for _event34 in ("glucose_finger", "map_arterial", "map_cuff"):
    _js34 = np.asarray(_groups31.get((_event34, "none"), [])[:24], dtype=int)
    _rs34 = [record_starts[(int(S0["stay_id"][j]),
             int(round(S0["hour"][j]*60)))][0] for j in _js34]
    _sp34 = NUM_SPEC[(_event34, "value")]
    _v34 = np.array([r.value for r in _rs34])
    _zb34 = np.clip((_v34-_sp34["median"])/_sp34["iqr"], -10, 10)
    _bin34 = np.searchsorted(_sp34["fine_bin_edges"], _v34, side="left")+1
    _bin34 = np.clip(_bin34, 1, 16)
    _base34 = torch.stack([recon_record_atom(r) for r in _rs34])
    _core34 = (_base34
               - torch.tensor(_zb34, dtype=torch.float32)[:, None]
                 * W0["embed.numeric.weight"][:, 0]
               - W0["embed.numeric.weight"][:, 1])
    _cent34 = head_centres[head_codes == CODE_ID[_event34]][0].numpy()
    _zc34 = np.clip((_cent34[_bin34]-_sp34["median"])/_sp34["iqr"], -10, 10)
    _candidates34 = [
        ("recorded", _zb34, 1), ("bin_centre", _zc34, 1),
        ("no_z", np.zeros_like(_zb34), 1),
        ("no_numeric", np.zeros_like(_zb34), 0),
        ("opposite_z", -_zb34, 1)
    ]
    _scores34 = []
    for _label34, _num34, _has34 in _candidates34:
        _aa34 = (_core34 + torch.tensor(_num34, dtype=torch.float32)[:, None]
                 * W0["embed.numeric.weight"][:, 0]
                 + _has34 * W0["embed.numeric.weight"][:, 1])
        _ee34 = recon_single_set(_aa34, prefix="encode")
        for _empty34 in ("zero", "encoded"):
            _uu34 = (torch.zeros((len(_js34), 192)) if _empty34 == "zero"
                     else recon_sets([[] for j in _js34],
                                     prefix="encode_action", empty_zero=False))
            _pr34 = _post33(_js34, _ee34, _uu34).numpy()
            _scores34.append((_label34, _empty34, round(float(np.sqrt(
                np.mean((_pr34-S0["x"][_js34, 960:])**2))), 6)))
    _value_tests34.append((_event34, sorted(_scores34, key=lambda x: x[2])[:4]))
print("=== MEASUREMENT NUMERIC TESTS ===")
print(_value_tests34)

with open("/work/notes.md", "a") as _f34:
    _f34.write(
        "\n## Run 34: corrected END routing and action-dose bottleneck\n"
        f"END tests: {_endtests34}; best convention {_best_end34}.\n"
        f"Measurement value tests: {_value_tests34}.\n"
        "Tested raw-dose and innovation recovery from incoming action vectors. "
        "No edits. Python runs: 34; forward passes: 3.\n"
    )
```