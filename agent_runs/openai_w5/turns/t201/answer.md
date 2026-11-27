The earlier checks establish that END atoms have no dose attributes and belong in the measurement encoder, not the action encoder. Adding them to the dose signal would be wrong. I’ll first fix the demonstrably incorrect memory boundaries and compare reconstruction settings. A substantial reduction in state error would support this fix; a remaining END-specific error would point to the world-vector builder.

```python
def wm_step_boundary176(j, include_current=False, age_mode="input", age_units="seconds"):
    sid = int(s4["stay_id"][j])
    start = int(np.flatnonzero(s4["stay_id"] == sid)[0])
    assert j > start
    prev = torch.from_numpy(s4["x"][j-1:j, 192:960]).reshape(1, 4, 192)
    dt = float(s4["hour"][j] - s4["hour"][j-1]) * 3600.0
    h = prev * torch.exp(-dt * torch.exp(W0["core.flow.a"]))
    key = (sid, int(round(float(s4["hour"][j]) * 60)))
    tab = groups4.get(key)
    atoms = [] if tab is None else [
        action_atom_exact(r.event, [r.rate], [r.dose])[0]
        for r in tab.itertuples(index=False)
        if r.event in ("insulin", "norepinephrine")]
    u = wm_set(torch.stack(atoms), "encode_action").reshape(1, 192) if atoms else torch.zeros(1, 192)
    z = torch.from_numpy(s4["x"][j:j+1, 960:992])
    tokens = torch.cat((h + W0["core.slots"],
                        wm_e4[j:j+1, None, :], u[:, None, :],
                        wm_lin(z, "z_in")[:, None, :]), dim=1)
    E = wm_norm(wm_layer(tokens, "core.prelude.0")[:, :4], "core.prelude_norm")
    end = j + int(include_current)
    first = max(start, end - 128)
    memory = wm_e4[first:end][None, :, :]
    divisor = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0}[age_units]
    ages = torch.as_tensor((s4["hour"][j] - s4["hour"][first:end]) * 3600.0 / divisor,
                           dtype=torch.float32).reshape(1, -1, 1)
    age = wm_mlp(torch.log1p(ages), "core.age")
    b = h + E
    for layer in range(2):
        b = wm_layer(b, f"core.blocks.0.{layer}", memory, age, age_mode)
    delta = F.softplus(W0["core.delta_in"])
    abar = torch.exp(-delta * torch.exp(W0["core.a_in"]))
    new_h = abar*h + delta*wm_lin(E, "core.b_in") + b - (h + E)
    return wm_layer(wm_lin(new_h, "core.c_out"), "core.coda.0")

ordinary176 = np.ones(192, dtype=bool)
ordinary176[[61, 21, 15, 66]] = False
old_memory176 = wm_memory_mode
trials176 = []
try:
    for mode in ("unnormalized", "norm_then_age", "age_then_norm", "age_then_plainnorm"):
        wm_memory_mode = mode
        for current in (False, True):
            for age_mode in ("input", "key"):
                for units in ("seconds", "hours"):
                    errs, perrs = [], []
                    for jj in (3, 585, 1110, 1675):
                        pred = wm_step_boundary176(jj, current, age_mode, units)[0].numpy()
                        truth = s4["x"][jj, 192:960].reshape(4, 192)
                        errs.append(float(np.sqrt(np.mean((pred[:, ordinary176]-truth[:, ordinary176])**2))))
                        perrs.append(float(np.sqrt(np.mean((pred[:, ~ordinary176]-truth[:, ~ordinary176])**2))))
                    trials176.append((np.mean(errs), np.mean(perrs), mode, current, age_mode, units, errs))
finally:
    wm_memory_mode = old_memory176
print("=== CORRECTED-BOUNDARY TRIALS (BEST 8) ===")
for result in sorted(trials176)[:8]:
    print(result)

empty_world176 = wm_set(torch.empty(0, 192), "encode")
print("\n=== WORLD VECTOR VS EMPTY SET ===")
for jj in (3, 585, 1110, 1675):
    print(jj, "RMSE", float(torch.sqrt(torch.mean((wm_e4[jj]-empty_world176)**2))))
print("\n=== ACTION ATOM IMPLEMENTATION ===")
compact_restart171(action_atom_exact, 3600)
```
