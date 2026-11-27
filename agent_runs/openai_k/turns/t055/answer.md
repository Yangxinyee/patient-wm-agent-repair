The sensitivity run stopped because the saved-state archive handle had been closed; it produced no model result. I’ll rerun using the retained metadata arrays. The test is unchanged: useful directions should move glucose or MAP reliably, with small effects on the other pair.

```python
import time
torch.set_num_threads(8)
r49_t0 = time.time()
r49_sensors = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
r49_stays, r49_hours = rr33_stays.copy(), rr33_hours.copy()
r49_register = np.full((len(r49_stays), 4, 2), np.nan)
r49_last = {}
for i49, (sid49, hour49) in enumerate(zip(r49_stays, r49_hours)):
    history49 = r49_last.setdefault(int(sid49), {})
    for row49, kind49 in bundle_records_audit[(int(sid49), float(hour49))]:
        if kind49 == 0 and row49.event in r49_sensors:
            history49[row49.event] = (float(row49.value), float(hour49))
    for j49, sensor49 in enumerate(r49_sensors):
        if sensor49 in history49:
            r49_register[i49, j49] = history49[sensor49]
r49_centres = torch.tensor(
    np.array([VOCAB["value_bin_centres"][s] for s in r49_sensors]),
    dtype=torch.float32)
r49_scales = np.array([num_spec0[(s, "value")]["iqr"] for s in r49_sensors])
r49_noise0 = np.random.default_rng(4901).normal(size=(8, 32)).astype(np.float32)
r49_noise = torch.tensor(np.concatenate([r49_noise0, -r49_noise0]))

def r49_readouts(out, h, indices):
    indices = np.asarray(indices, dtype=int)
    hours49 = r49_hours[indices]
    dt_np49 = np.maximum(np.ceil(hours49 + 1e-5)-hours49, 1e-4)*3600
    dt49 = torch.tensor(dt_np49, dtype=torch.float32)
    flowed49 = h * torch.exp(-dt49[:, None, None]*torch.exp(W0["core.flow.a"]))
    ctx49 = out + norm0(flowed49.mean(1), "core.out_norm")
    ctx49 += mlp0(torch.log(dt49)[:, None]/10, "gap")
    mu49, ls49 = mlp0(ctx49, "z_prior").chunk(2, -1)
    zs49 = mu49[:, None, :] + torch.exp(ls49)[:, None, :]*r49_noise[None]
    q49 = norm0(ctx49[:, None, :] + lin0(zs49, "z_head")
                + W0["type_emb.weight"][1], "query_norm")
    readings49 = []
    for j49, sensor49 in enumerate(r49_sensors):
        spec49 = num_spec0[(sensor49, "value")]
        lv49, lh49 = r49_register[indices, j49].T
        has49 = np.isfinite(lv49)
        value49 = np.where(has49, lv49, spec49["median"])
        bins49 = np.where(has49, 1 + np.searchsorted(
            spec49["fine_bin_edges"], value49, side="left"), 0)
        numeric49 = np.zeros((len(indices), 8), dtype=np.float32)
        numeric49[:, 0] = has49
        age49 = np.where(has49, np.maximum(
            hours49 + dt_np49/3600-lh49, 0)*3600, 0)
        numeric49[:, 1] = np.log1p(age49)/10
        numeric49[:, 2] = np.where(has49, np.clip(
            (value49-spec49["median"])/spec49["iqr"], -8, 8), 0)
        numeric49[:, 3] = has49
        code49 = code_id0[sensor49]
        bins49 = torch.tensor(bins49)
        wc49 = (q49 + W0["embed.code.weight"][code49]
                + lin0(torch.tensor(numeric49), "last_numeric")[:, None, :]
                + W0["embed.bins.value.weight"][bins49][:, None, :]
                + W0["embed.bins.rate.weight"][0]
                + W0["last_status.weight"][0])
        logits49 = mlp0(wc49, "attr_heads.value_bin")
        logits49 += W0["embed_transition.value_bin.weight"][code49*17+bins49][:, None, :]
        readings49.append((torch.softmax(logits49, -1) *
                           r49_centres[j49]).sum(-1).mean(1))
    return torch.stack(readings49, 1)

r49_h0, r49_out0 = repaired_core_local(
    core_probe_h, core_probe_e, core_probe_u, core_probe_z,
    core_probe_memory, core_probe_mask, core_probe_ages)
r49_mean0 = r49_readouts(r49_out0, r49_h0, core_probe_indices)
r49_eps = 0.02
r49_J = np.zeros((len(core_probe_indices), 4, 32))
for j49 in range(32):
    zp49, zm49 = core_probe_z.clone(), core_probe_z.clone()
    zp49[:, j49] += r49_eps
    zm49[:, j49] -= r49_eps
    hp49, op49 = repaired_core_local(
        core_probe_h, core_probe_e, core_probe_u, zp49,
        core_probe_memory, core_probe_mask, core_probe_ages)
    hm49, om49 = repaired_core_local(
        core_probe_h, core_probe_e, core_probe_u, zm49,
        core_probe_memory, core_probe_mask, core_probe_ages)
    r49_J[:, :, j49] = ((r49_readouts(op49, hp49, core_probe_indices)
                        - r49_readouts(om49, hm49, core_probe_indices))
                       / (2*r49_eps)).numpy()
r49_targets = np.array([[-1.7, -1.7, 0, 0], [0, 0, 2.8, 2.8]]).T
r49_A = (r49_J/r49_scales[None, :, None]).reshape(-1, 32)
r49_B = np.tile(r49_targets/r49_scales[:, None], (len(core_probe_indices), 1))
r49_directions = {}
print("Baseline sensor means:", r49_mean0.mean(0).numpy())
print("Mean sensitivity singular values:", np.linalg.svd(r49_J.mean(0), compute_uv=False))
for alpha49 in [1e-5, 1e-4, 1e-3, 1e-2, 0.1]:
    K49 = np.linalg.solve(r49_A.T @ r49_A + alpha49*np.eye(32), r49_A.T @ r49_B)
    r49_directions[alpha49] = K49
    response49 = np.einsum("nsz,zd->nsd", r49_J, K49)
    print("ridge", alpha49, "latent norms", np.linalg.norm(K49, axis=0),
          "mean response", response49.mean(0).round(3).tolist(),
          "response SD", response49.std(0).round(3).tolist())
print("Elapsed seconds:", round(time.time()-r49_t0, 1))
with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 48–50: dose decoder and latent sensitivities\n"
            "- Decoder outputs are rate_I, rate_NE, presence_I, presence_NE; empty-action outputs are near zero.\n"
            "- Run 48 stopped on a diagnostic key error; run 49 stopped on a closed NPZ handle.\n"
            "- Run 50 uses retained metadata and measures four-sensor posterior-latent sensitivities.\n"
            "- No tensors edited. Python runs: 50; forward passes: 4.\n")
```