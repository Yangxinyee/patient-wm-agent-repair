Changing insulin rate moves the original posterior substantially, but norepinephrine’s posterior movement is tiny. Rate decoding is only moderately accurate, and the empty-END hypothesis fails. Amplifying existing dose directions alone therefore looks unreliable, especially for norepinephrine. I’ll inspect the recurrent layers and validate measurement encoding next, so we can test latent directions that actually move the intended reading.

```python
print("=== CORE TENSORS ===")
for _name in NAMES:
    if _name.startswith("core."):
        print(_name.removeprefix("core."), SHAPES[_name])

print("\n=== ENCODER HELPERS ===")
for _name in ["latent_encode_batch", "repair_mlp_act", "repair_norm"]:
    _fn = globals()[_name]
    print(_name, "locals:", _fn.__code__.co_varnames,
          "globals:", _fn.__code__.co_names,
          "constants:", [v for v in _fn.__code__.co_consts
                         if isinstance(v, (str, int, float, tuple))][:15])
print("Numeric spec keys:", list(numeric_specs)[:8])
print("One spec:", summary49(next(iter(numeric_specs.values())), 700))

obs_lookup52 = {}
for (_s, _t), _g in latent_records.groupby(["stay_id", "hour"], sort=False):
    obs_lookup52[(int(_s), round(float(_t), 4))] = _g.to_dict("records")

print("\n=== SINGLE MEASUREMENT POSTERIOR CHECK ===")
obs_samples52 = {}
for _sensor in MEAS:
    _samples = []
    for _i in range(1, len(latent_sid)):
        _key = (int(latent_sid[_i]), round(float(latent_hour[_i]), 4))
        _rows = obs_lookup52.get(_key, [])
        if (latent_sid[_i-1] == latent_sid[_i]
                and len(_rows) == 1 and _rows[0]["event"] == _sensor
                and not latent_ends.get(_key, [])):
            _samples.append((_i, _rows[0]))
        if len(_samples) == 20:
            break
    obs_samples52[_sensor] = _samples
    if not _samples:
        print(_sensor, "no single-measurement examples")
        continue
    _ix = np.array([i for i, r in _samples])
    _ctx = latent_context(_ix)
    _base = torch.stack([repair_atom_base(r, kind=0, include_numbers=True)
                         for i, r in _samples])
    _scores = []
    for _unit in range(repair_weights["embed.categorical.unit.weight"].shape[0]):
        _atoms = _base + repair_weights["embed.categorical.unit.weight"][_unit]
        _e = latent_encode_batch(_atoms, "encode")
        _z = latent_post_mean(_ctx, _e, torch.zeros(len(_ix), 192))
        _rms = float(torch.sqrt(torch.mean(
            (_z-torch.from_numpy(latent_state[_ix, -32:]))**2)))
        _scores.append((_rms, _unit))
    print(_sensor, "n=", len(_samples), "best units:", sorted(_scores)[:2])

with Path("/work/notes.md").open("a") as _f:
    _f.write("\n## Python 52; forward 6 — posterior diagnosis\n"
             "- Clean-start posterior reconstruction RMS 0.000364; accurate enough for local probes, not bit-exact.\n"
             "- Insulin doubling strongly moves posterior (dose-direction norm 1.10); NE movement is tiny (0.030).\n"
             "- Existing action-vector rate decoding held-out R2 ~0.75 insulin, ~0.68 NE; empty-END hypothesis fails (RMS 0.132).\n"
             "- Simple amplification is not yet justified. Inspecting physical output directions via recurrent computations. No edits.\n")
```
