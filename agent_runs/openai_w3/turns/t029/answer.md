The two drug-atom builders differ because one omits metadata embeddings. I’ll now test explicit metadata and bin conventions against the stored posterior states. A near-zero error across multiple readings would identify the encoding; a persistent error would refute this explanation.

```python
import itertools, math
torch.set_num_threads(8)

def state_context(indices, gap_mode="log1p"):
    indices = np.asarray(indices, dtype=int)
    prev = indices - 1
    dt_s = (np.asarray(S["hour"])[indices] - np.asarray(S["hour"])[prev]) * 3600.0
    xx = torch.as_tensor(np.asarray(S["x"]), dtype=torch.float32)
    h = xx[prev, 192:960].reshape(-1, 4, 192)
    flowed = h * torch.exp(
        -torch.as_tensor(dt_s, dtype=torch.float32)[:, None, None]
        * torch.exp(W["core.flow.a"])[None, None, :]
    )
    gap_x = np.log1p(dt_s) if gap_mode == "log1p" else np.log(np.maximum(dt_s, 1.0))
    return (xx[prev, :192] + layer_norm(flowed.mean(1), "core.out_norm")
            + mlp(torch.as_tensor(gap_x[:, None] / 10, dtype=torch.float32), "gap"))

def isolated_record(i, event):
    sid, hour = int(S["stay_id"][i]), float(S["hour"][i])
    rows_here = state_records[
        (state_records.stay_id == sid) &
        (np.abs(state_records.hour - hour) < 0.0002) &
        (state_records.event == event)
    ]
    return next(rows_here.itertuples(index=False))

def posterior_mean(context, e, u):
    return mlp(torch.cat([context, e, u], dim=1), "z_post")[:, :32]

print("=== CLEAN CONTEXT / ATOM CHECK ===")
for event in ("insulin", "norepinephrine", "map_cuff", "glucose_finger"):
    idx = np.asarray(isolated_by_event[event][:12], dtype=int)
    aa = torch.stack([make_atom(isolated_record(i, event)) for i in idx])
    target = torch.as_tensor(np.asarray(S["x"])[idx, 960:], dtype=torch.float32)
    if event in ("insulin", "norepinephrine"):
        e = e_empty.expand(len(idx), -1)
        u = encode_one_set(aa, "encode_action")
    else:
        e = encode_one_set(aa, "encode")
        u = torch.zeros_like(e)
    print(event, {
        mode: float(torch.mean((posterior_mean(state_context(idx, mode), e, u)-target)**2))
        for mode in ("log1p", "log")
    })

extra_names = [
    "embed.categorical.action.weight",
    "embed.categorical.route.weight",
    "embed.categorical.status.weight",
    "embed.categorical.discharged_to.weight",
    "embed.text.weight",
]
extra_vectors = torch.stack([W[name][1] - W[name][0] for name in extra_names])
configs = list(itertools.product(range(6), range(4), ("left", "right"), range(32)))
metadata_search = {}

print("\n=== MEASUREMENT METADATA SEARCH ===")
for event in ("map_cuff", "map_arterial", "glucose_finger", "glucose_lab"):
    idx = np.asarray(isolated_by_event[event][:8], dtype=int)
    records = [isolated_record(i, event) for i in idx]
    atoms_base = torch.stack([make_atom(r) for r in records])
    context = state_context(idx)
    target = torch.as_tensor(np.asarray(S["x"])[idx, 960:], dtype=torch.float32)
    original_unit = 2 if event.startswith("map_") else 5
    edges = np.asarray(numeric_tables[(event, "value")]["fine_bin_edges"])
    bins_left = np.array([1 + np.sum(r.value > edges) for r in records])
    bins_right = np.array([1 + np.sum(r.value >= edges) for r in records])
    right_delta = W["embed.bins.value.weight"][bins_right] - W["embed.bins.value.weight"][bins_left]
    scores = []
    for lo in range(0, len(configs), 128):
        cc = configs[lo:lo+128]
        aa = atoms_base[None, :, :].repeat(len(cc), 1, 1)
        for j, (unit, kq, side, bits) in enumerate(cc):
            aa[j] += (W["embed.categorical.unit.weight"][unit]
                      - W["embed.categorical.unit.weight"][original_unit]
                      + W["embed.kq.weight"][kq] - W["embed.kq.weight"][0])
            if side == "right":
                aa[j] += right_delta
            for bit in range(5):
                if bits & (1 << bit):
                    aa[j] += extra_vectors[bit]
        ee = encode_one_set(aa.reshape(-1, 192), "encode")
        zz = posterior_mean(context.repeat(len(cc), 1), ee, torch.zeros_like(ee))
        error = ((zz.reshape(len(cc), len(idx), 32) - target[None])**2).mean((1, 2))
        scores.extend((float(error[j]), cc[j]) for j in range(len(cc)))
    metadata_search[event] = sorted(scores)
    print(event, metadata_search[event][:4])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 25; forwards 3. Clean context checks and measurement metadata enumeration (unit, kq, bin equality, optional categorical fields); no edits.\n")
```