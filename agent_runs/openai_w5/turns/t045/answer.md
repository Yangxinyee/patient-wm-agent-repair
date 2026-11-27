We have two independent training-state samples and exact encoders for the main event types. I’ll now compare reconstructed drug-bundle transitions with the saved GPU transitions. Small errors would justify using local perturbations to design an edit; large errors would mean a recurrence convention is still missing.

```python
print("Measurement units:", measurement_unit_local)
print("Event kinds:", VOCAB["event_kinds"])
print("Atom kinds:", VOCAB["atom_kinds"])
for nm in ["measurement_atom_local", "action_atom_exact"]:
    fun = globals()[nm]
    print(nm, "constants:", [c for c in fun.__code__.co_consts
                             if isinstance(c, (str, int, float, tuple))])
for nm in ["empty_measure_local", "empty_action_local"]:
    vv = globals()[nm]
    print(nm, type(vv).__name__, tuple(vv.shape), float(vv.norm()))

def wm_set(atoms, prefix):
    n = len(atoms)
    agg = wm_mlp(atoms, prefix + ".mlp1").sum(0, keepdim=True) / np.sqrt(max(n, 1))
    size = torch.tensor([[np.log1p(n)]], dtype=torch.float32)
    return wm_norm(wm_mlp(agg + wm_lin(size, prefix + ".size"),
                          prefix + ".mlp2"), prefix + ".norm")[0]

s4 = state4
records4 = train[train.stay_id.isin(np.unique(s4["stay_id"]))].copy()
records4["minute"] = np.rint(records4.hour * 60).astype(int)
groups4 = {(int(sid0), int(minute0)): group
           for (sid0, minute0), group in records4.groupby(["stay_id", "minute"])}
wm_e4 = []
for j in range(len(s4["x"])):
    key4 = (int(s4["stay_id"][j]), int(round(s4["hour"][j] * 60)))
    group4 = groups4.get(key4)
    atoms4 = []
    if group4 is not None:
        for rr4 in group4.itertuples(index=False):
            if rr4.event in measurement_unit_local:
                atoms4.append(measurement_atom_local(
                    rr4.event, [rr4.value], measurement_unit_local[rr4.event])[0])
    wm_e4.append(wm_set(torch.stack(atoms4) if atoms4 else torch.empty(0, 192), "encode"))
wm_e4 = torch.stack(wm_e4)

wm_drug_rows = np.flatnonzero(s4["has_insulin"] | s4["has_norepinephrine"])
wm_drug_rows = wm_drug_rows[
    (wm_drug_rows > 0) &
    (s4["stay_id"][wm_drug_rows] == s4["stay_id"][np.maximum(wm_drug_rows - 1, 0)])
][:24]

def wm_step_saved(j, include_current=False, age_mode="input", age_units="seconds"):
    ctx4 = context_from_saved(np.array([j]), s4)
    # Use the saved latent first, isolating recurrence reconstruction.
    z4 = torch.from_numpy(s4["x"][j:j+1, 960:992])
    prev4 = torch.from_numpy(s4["x"][j-1:j, 192:960]).reshape(1, 4, 192)
    dt4 = float((s4["hour"][j] - s4["hour"][j-1]) * 3600)
    h4 = prev4 * torch.exp(-dt4 * torch.exp(W0["core.flow.a"]))
    key4 = (int(s4["stay_id"][j]), int(round(s4["hour"][j] * 60)))
    group4 = groups4[key4]
    aa4 = [action_atom_exact(rr4.event, [rr4.rate], [rr4.dose])[0]
           for rr4 in group4.itertuples(index=False)
           if rr4.event in ("insulin", "norepinephrine")]
    u4 = wm_set(torch.stack(aa4), "encode_action").reshape(1, 192)
    tok4 = torch.cat([h4 + W0["core.slots"],
                      wm_e4[j:j+1, None, :], u4[:, None, :],
                      wm_lin(z4, "z_in")[:, None, :]], dim=1)
    E4 = wm_norm(wm_layer(tok4, "core.prelude.0")[:, :4], "core.prelude_norm")
    start4 = np.searchsorted(s4["stay_id"], s4["stay_id"][j])
    end4 = j + int(include_current)
    mem4 = wm_e4[max(start4, end4-128):end4][None, :, :]
    divisor4 = {"seconds": 1., "minutes": 60., "hours": 3600.}[age_units]
    ages4 = torch.tensor(
        (s4["hour"][j] - s4["hour"][max(start4, end4-128):end4]) * 3600 / divisor4,
        dtype=torch.float32).reshape(1, -1, 1)
    age4 = wm_mlp(torch.log1p(ages4), "core.age")
    b4 = h4 + E4
    for layer4 in range(2):
        b4 = wm_layer(b4, f"core.blocks.0.{layer4}", mem4, age4, age_mode)
    delta4 = F.softplus(W0["core.delta_in"])
    abar4 = torch.exp(-delta4 * torch.exp(W0["core.a_in"]))
    new4 = abar4 * h4 + delta4 * wm_lin(E4, "core.b_in") + b4 - (h4 + E4)
    new4 = wm_layer(wm_lin(new4, "core.c_out"), "core.coda.0")
    return new4

wm_transition_trials = []
for inc4 in [False, True]:
    for am4 in ["input", "key"]:
        for au4 in ["seconds", "minutes", "hours"]:
            errors4 = []
            for j4 in wm_drug_rows[:8]:
                pred4 = wm_step_saved(int(j4), inc4, am4, au4)
                target4 = torch.from_numpy(s4["x"][j4, 192:960]).reshape(1, 4, 192)
                errors4.append(float((pred4 - target4).square().mean().sqrt()))
            wm_transition_trials.append((float(np.mean(errors4)), inc4, am4, au4))
print("Transition trials:", sorted(wm_transition_trials))
```
