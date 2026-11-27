Changing metadata did not explain the measurement mismatch, so that hypothesis is unsupported. I’ll test time-flow and numerical-input conventions, and separately decode infusion-end events. These checks matter because most ongoing doses share a bundle with the previous infusion’s end; near-zero posterior error would confirm a usable action reconstruction.

```python
def context_with_flow(indices, factor):
    indices = np.asarray(indices, dtype=int)
    prev = indices - 1
    dt_s = (np.asarray(S["hour"])[indices] - np.asarray(S["hour"])[prev]) * 3600
    xx = torch.as_tensor(np.asarray(S["x"]), dtype=torch.float32)
    h = xx[prev, 192:960].reshape(-1, 4, 192)
    h = h * torch.exp(
        -factor * torch.as_tensor(dt_s, dtype=torch.float32)[:, None, None]
        * torch.exp(W["core.flow.a"])[None, None, :]
    )
    return (xx[prev, :192] + layer_norm(h.mean(1), "core.out_norm")
            + mlp(torch.as_tensor(np.log1p(dt_s)[:, None]/10, dtype=torch.float32), "gap"))

print("=== NUMERICAL / FLOW CHECK ===")
numerical_search = {}
for event in ("map_cuff", "glucose_finger", "glucose_lab", "insulin"):
    idx = np.asarray(isolated_by_event[event][:10], dtype=int)
    records = [isolated_record(i, event) for i in idx]
    aa = torch.stack([make_atom(r) for r in records])
    target = torch.as_tensor(np.asarray(S["x"])[idx, 960:], dtype=torch.float32)
    is_drug = event == "insulin"
    scores = []
    unit_options = [3] if is_drug else ([2] if event.startswith("map_") else [1, 5])
    for unit, omit_kq, flow, pool_scale in itertools.product(
            unit_options, (False, True), (0.0, 1.0, 2.0), (0.5, 1.0, 2**-0.5, 2**0.5, 2.0)):
        bb = aa.clone()
        original_unit = 3 if is_drug else (2 if event.startswith("map_") else 5)
        bb += W["embed.categorical.unit.weight"][unit] - W["embed.categorical.unit.weight"][original_unit]
        if omit_kq:
            bb -= W["embed.kq.weight"][0]
        variants = {"raw": bb}
        if not is_drug:
            table = numeric_tables[(event, "value")]
            edges = np.asarray(table["fine_bin_edges"])
            values = np.asarray([r.value for r in records])
            bins = np.asarray([1 + np.sum(v > edges) for v in values])
            centres = np.asarray(VOCAB["value_bin_centres"][event])
            zraw = np.clip((values-table["median"])/table["iqr"], -10, 10)
            zcentre = np.clip((centres[bins]-table["median"])/table["iqr"], -10, 10)
            variants["bin_centres"] = bb + torch.as_tensor(
                zcentre-zraw, dtype=torch.float32)[:, None] * W["embed.numeric.weight"][:, 0]
            variants["no_z_value"] = bb - torch.as_tensor(
                zraw, dtype=torch.float32)[:, None] * W["embed.numeric.weight"][:, 0]
            variants["no_numeric_value"] = variants["no_z_value"] - W["embed.numeric.weight"][:, 1]
        for vname, atoms_v in variants.items():
            prefix = "encode_action" if is_drug else "encode"
            pooled = mlp(atoms_v, prefix + ".mlp1") * pool_scale
            pooled += linear(torch.full((len(idx), 1), math.log(2)), prefix + ".size")
            vec = layer_norm(mlp(pooled, prefix + ".mlp2"), prefix + ".norm")
            e = e_empty.expand(len(idx), -1) if is_drug else vec
            u = vec if is_drug else torch.zeros_like(vec)
            zz = posterior_mean(context_with_flow(idx, flow), e, u)
            scores.append((float(((zz-target)**2).mean()),
                           (unit, omit_kq, flow, pool_scale, vname)))
    numerical_search[event] = sorted(scores)
    print(event, numerical_search[event][:3])

print("\n=== END-ATOM SEARCH ===")
end_search = {}
end_configs = list(itertools.product(
    range(3), range(5), range(6), range(4), ("clear", "preserve"), (False, True),
    ("action", "observation")))
for event in ("insulin", "norepinephrine"):
    idx = np.asarray(isolated_by_event[event + " END"][:8], dtype=int)
    records = []
    for i in idx:
        rr = state_records[
            (state_records.stay_id == int(S["stay_id"][i])) &
            (state_records.event == event) &
            (np.abs(state_records.end_hour-float(S["hour"][i])) < 0.0002)
        ]
        records.append(next(rr.itertuples(index=False)))
    clear_atoms = torch.stack([make_atom(r, True, "clear_all") for r in records])
    full_atoms = torch.stack([make_atom(r, True, "preserve") for r in records])
    context = context_with_flow(idx, 1.0)
    target = torch.as_tensor(np.asarray(S["x"])[idx, 960:], dtype=torch.float32)
    orig_unit, orig_rateunit = ((3, 2) if event == "insulin" else (4, 3))
    scores = []
    for lo in range(0, len(end_configs), 128):
        cc = end_configs[lo:lo+128]
        aa = torch.empty((len(cc), len(idx), 192))
        for j, (kind, ek, unit, ru, mode, omit, route) in enumerate(cc):
            aa[j] = clear_atoms if mode == "clear" else full_atoms
            aa[j] += W["embed.kind.weight"][kind] - W["embed.kind.weight"][2]
            aa[j] += W["embed.ekind.weight"][ek] - W["embed.ekind.weight"][2]
            aa[j] += W["embed.categorical.unit.weight"][unit] - W["embed.categorical.unit.weight"][0 if mode == "clear" else orig_unit]
            aa[j] += W["embed.categorical.rate_unit.weight"][ru] - W["embed.categorical.rate_unit.weight"][0 if mode == "clear" else orig_rateunit]
            if omit:
                aa[j] -= W["embed.kq.weight"][0]
        flat = aa.reshape(-1, 192)
        av = encode_one_set(flat, "encode_action")
        ov = encode_one_set(flat, "encode")
        ee = e_empty.expand(len(flat), -1).clone()
        uu = av.clone()
        for j, cfg in enumerate(cc):
            if cfg[-1] == "observation":
                sl = slice(j*len(idx), (j+1)*len(idx))
                ee[sl], uu[sl] = ov[sl], 0
        zz = posterior_mean(context.repeat(len(cc), 1), ee, uu)
        error = ((zz.reshape(len(cc), len(idx), 32)-target[None])**2).mean((1, 2))
        scores.extend((float(error[j]), cc[j]) for j in range(len(cc)))
    end_search[event] = sorted(scores)
    print(event, end_search[event][:3])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 26; forwards 3. Metadata-only explanation rejected. Testing flow/input scaling and END conventions; no weights changed.\n")
```