The saved state layout and insulin encoding are now explicit. I’ll verify the measurement encoding directly, including its categorical fields, and check how record rows correspond to bundles. A posterior match near numerical precision would confirm the settings; a larger error would stop the transition experiment before any weights change.

```python
import time, itertools
t123 = time.time()

def tensor_prefix(ending):
    hits = [n[:-len(".weight")] for n in NAMES if n.endswith(ending + ".weight")]
    if len(hits) != 1:
        raise ValueError((ending, hits))
    return hits[0]

rec_outnorm = tensor_prefix("out_norm")
rec_gap = tensor_prefix("gap.0")[:-2]
rec_zpost = tensor_prefix("z_post.0")[:-2]
rec_zin = tensor_prefix("z_in")
print("PREFIXES:", rec_outnorm, rec_gap, rec_zpost, rec_zin)

def rec_context(prev, dt_seconds):
    prev = torch.as_tensor(prev, dtype=torch.float32)
    dt_seconds = torch.as_tensor(dt_seconds, dtype=torch.float32).reshape(-1, 1)
    hf = prev[:, 192:960].reshape(-1, 4, 192) * torch.exp(
        -dt_seconds[:, None, :] * torch.exp(rm_W["core.flow.a"])[None, None, :])
    ctx = (prev[:, :192] + rm_ln(hf.mean(1), rec_outnorm)
           + rm_mlp(torch.log(dt_seconds.clamp_min(1e-8)) / 10, rec_gap))
    return ctx, hf

rec_mc, rec_mhf = rec_context(r118_prev, r118_dt)
rec_v = torch.as_tensor(r118_values, dtype=torch.float32)
rec_num = torch.zeros((len(rec_v), 2, 6))
rec_bin = torch.zeros((len(rec_v), 2), dtype=torch.long)
for j, ev in enumerate(("glucose_finger", "map_arterial")):
    ns = rm_normspec[(ev, "value")]
    rec_num[:, j, 0] = ((rec_v[:, j] - ns["median"]) / ns["iqr"]).clamp(-10, 10)
    rec_num[:, j, 1] = 1
    rec_bin[:, j] = torch.as_tensor(
        np.searchsorted(ns["fine_bin_edges"], rec_v[:, j].numpy(), side="left") + 1)
rec_base = (rm_W["embed.code.weight"][torch.tensor([6, 7])][None]
            + rm_W["embed.kind.weight"][0]
            + rm_W["embed.ekind.weight"][1]
            + rm_W["embed.bins.value.weight"][rec_bin]
            + rm_lin(rec_num, "embed.numeric"))
rec_measurement_checks = []
for gu, mu, kq in itertools.product(range(6), range(6), [-1, 0, 1, 2, 3]):
    av = rec_base.clone()
    av[:, 0] += rm_W["embed.categorical.unit.weight"][gu]
    av[:, 1] += rm_W["embed.categorical.unit.weight"][mu]
    if kq >= 0:
        av += rm_W["embed.kq.weight"][kq]
    ev = rm_set(av, "encode")
    zv = rm_mlp(torch.cat([rec_mc, ev, r118_u_options["empty_encoder"]], -1), rec_zpost)[:, :32]
    error = float(torch.mean((zv - r118_target) ** 2).sqrt())
    rec_measurement_checks.append((error, gu, mu, kq))
rec_measurement_checks.sort()
print("MEASUREMENT CHECKS:", rec_measurement_checks[:8])
print("MEASUREMENT DT RANGE:", float(torch.as_tensor(r118_dt).min()),
      float(torch.as_tensor(r118_dt).max()))
print("FIRST BUNDLES:")
for i in range(14):
    sid, hr = int(rm_states["stay_id"][i]), float(rm_states["hour"][i])
    evs = [key[2] for key in rm_lookup if key[:2] == (sid, hr)]
    print(i, hr, int(rm_states["n_atoms"][i]), evs)
print("SECONDS:", round(time.time() - t123, 2))
```