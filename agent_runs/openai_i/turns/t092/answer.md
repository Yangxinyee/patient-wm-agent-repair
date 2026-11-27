The exposed states cover 100 training stays, and the earlier exact tests provide sensor-specific encodings that differ from the metadata-based helper. I’ll rebuild every bundle with those encodings, then test infusion-end fields and minute-rounded timing together. Near–floating-point posterior agreement would make the hidden features usable for an edit; a larger error would refute the reconstruction.

```python
import itertools, time
torch.set_num_threads(4)
t84 = time.time()

unit84 = {"glucose_finger": 0, "glucose_lab": 5,
          "map_arterial": 2, "map_cuff": 2,
          "insulin": 3, "norepinephrine": 4}
rateunit84 = {"insulin": 2, "norepinephrine": 3}
ids84 = set(S83["stay_id"].tolist())
records84 = train[train.stay_id.isin(ids84)].copy()
lookup84 = {(int(s), int(round(float(h)*60))): i
            for i, (s, h) in enumerate(zip(S83["stay_id"], S83["hour"]))}
B84 = [[] for _ in range(len(S83["hour"]))]
for r84 in records84.to_dict("records"):
    k84 = (int(r84["stay_id"]), int(round(float(r84["hour"])*60)))
    if k84 in lookup84:
        B84[lookup84[k84]].append((r84, False))
    if pd.notna(r84["end_hour"]):
        k84 = (int(r84["stay_id"]), int(round(float(r84["end_hour"])*60)))
        if k84 in lookup84:
            B84[lookup84[k84]].append((r84, True))

nb84 = len(B84)
counts84 = np.array([len(b) for b in B84])
endcount84 = np.array([sum(end for _, end in b) for b in B84])
actioncount84 = np.array([sum(not end and r["event"] in rateunit84
                             for r, end in b) for b in B84])
obscount84 = counts84 - actioncount84
valid84 = np.r_[False, S83["stay_id"][1:] == S83["stay_id"][:-1]]
print("Atom-count mismatches:", int(np.sum(counts84 != S83["n_atoms"])))
print("Bundles:", nb84, "valid:", valid84.sum(),
      "with END:", (endcount84 > 0).sum())

def atom84(row, end=False, end_unit="absent", end_rateunit="absent",
           end_numbers=False, end_kind=2, end_ekind="same"):
    ev = row["event"]
    uk = unit84.get(ev, 0)
    rk = rateunit84.get(ev, 0)
    ek = ekind_ids57[event_info57[ev]["event_kind"]]
    if end:
        uk = {"absent": 0, "same": uk, "one": 1}[end_unit]
        rk = {"absent": 0, "same": rk, "one": 1}[end_rateunit]
        if end_ekind == "zero":
            ek = 0
    return complete_atom57(row, kind=end_kind if end else 0, ekind=ek,
                           unit=uk, rate_unit=rk,
                           numbers=end_numbers if end else True)

def pooled84(atom_vectors, bundle_indices, counts, prefix):
    sums = torch.zeros((nb84, 192), dtype=torch.float32)
    if atom_vectors:
        atoms = torch.stack(atom_vectors)
        encoded = repair_mlp_act(atoms, prefix+".mlp1", "gelu")
        sums.index_add_(0, torch.tensor(bundle_indices, dtype=torch.long), encoded)
    n = torch.tensor(counts, dtype=torch.float32).reshape(-1, 1)
    sums = sums / torch.sqrt(n.clamp_min(1))
    sums += repair_linear(torch.log1p(n), prefix+".size")
    return repair_norm(repair_mlp_act(sums, prefix+".mlp2", "gelu"),
                       prefix+".norm")

world_atoms84, world_indices84, action_atoms84, action_indices84 = [], [], [], []
end_rows84, end_indices84 = [], []
for i84, b84 in enumerate(B84):
    for r84, end84 in b84:
        if end84:
            end_rows84.append(r84)
            end_indices84.append(i84)
        elif r84["event"] in rateunit84:
            action_atoms84.append(atom84(r84))
            action_indices84.append(i84)
        else:
            world_atoms84.append(atom84(r84))
            world_indices84.append(i84)

Ubase84 = pooled84(action_atoms84, action_indices84, actioncount84, "encode_action")
Eworld84 = pooled84(world_atoms84, world_indices84,
                   obscount84-endcount84, "encode")
X84 = torch.from_numpy(S83["x"].astype(np.float32))
Ztarget84 = X84[:, 960:992]
prevout84 = torch.roll(X84[:, :192], 1, 0)
prevh84 = torch.roll(X84[:, 192:960].reshape(-1, 4, 192), 1, 0)

def context84(rounded=True, gap_log1p=False):
    hours = S83["hour"].astype(float)
    times = np.rint(hours*60)*60 if rounded else hours*3600
    dt = np.r_[3600., np.diff(times)]
    dt[~valid84] = 3600.
    dt_t = torch.tensor(dt, dtype=torch.float32)
    flowed = prevh84 * torch.exp(
        -dt_t[:, None, None] * torch.exp(repair_weights["core.flow.a"])[None, None, :])
    g = torch.log1p(dt_t) if gap_log1p else torch.log(dt_t.clamp_min(1))
    return (prevout84 + repair_norm(flowed.mean(1), "core.out_norm")
            + repair_mlp_act(g[:, None]/10, "gap", "gelu"))

def posterior84(ctx, e, u):
    f = torch.nn.functional.gelu(
        repair_linear(torch.cat([ctx, e, u], 1), "z_post.0"))
    z = repair_linear(f, "z_post.2")[:, :32]
    return f, z

clean84 = torch.from_numpy(valid84 & (endcount84 == 0))
clean_results84 = []
for rounded84, log1p84, empty_e84, empty_u84 in itertools.product(
        [True, False], [False, True], ["encoded", "zero"], ["encoded", "zero"]):
    c84 = context84(rounded84, log1p84)
    e84, u84 = Eworld84.clone(), Ubase84.clone()
    if empty_e84 == "zero":
        e84[obscount84 == 0] = 0
    if empty_u84 == "zero":
        u84[actioncount84 == 0] = 0
    _, z84 = posterior84(c84, e84, u84)
    rms84 = float(torch.sqrt(torch.mean((z84[clean84]-Ztarget84[clean84])**2)))
    clean_results84.append((rms84, rounded84, log1p84, empty_e84, empty_u84))
clean_results84.sort()
print("Best clean conventions:", clean_results84[:3])
_, rounded84, log1p84, empty_e84, empty_u84 = clean_results84[0]
C84 = context84(rounded84, log1p84)
U84 = Ubase84.clone()
if empty_u84 == "zero":
    U84[actioncount84 == 0] = 0

end_results84 = []
best_rms84 = np.inf
mask84 = torch.from_numpy(valid84)
for eu84, er84, en84, ek84, ee84 in itertools.product(
        ["absent", "same", "one"], ["absent", "same", "one"],
        [False, True], [2, 0, 1], ["same", "zero"]):
    end_atoms84 = [atom84(r, True, eu84, er84, en84, ek84, ee84)
                   for r in end_rows84]
    e84 = pooled84(world_atoms84 + end_atoms84,
                  world_indices84 + end_indices84, obscount84, "encode")
    if empty_e84 == "zero":
        e84[obscount84 == 0] = 0
    f84, z84 = posterior84(C84, e84, U84)
    rms84 = float(torch.sqrt(torch.mean((z84[mask84]-Ztarget84[mask84])**2)))
    cfg84 = (eu84, er84, en84, ek84, ee84)
    end_results84.append((rms84, cfg84))
    if rms84 < best_rms84:
        best_rms84 = rms84
        E84, F84, Z84, end_config84 = e84.clone(), f84.clone(), z84.clone(), cfg84
end_results84.sort()
print("Best END conventions:", end_results84[:5])
err84 = torch.sqrt(torch.mean((Z84-Ztarget84)**2, 1)).numpy()
for label84, m84 in [
    ("all", valid84), ("actions", valid84 & (actioncount84 > 0)),
    ("reading/visit only", valid84 & (actioncount84 == 0) & (endcount84 == 0)),
    ("END present", valid84 & (endcount84 > 0))]:
    print(label84, "n", int(m84.sum()),
          "mean/p95/max RMS", np.quantile(err84[m84], [0.5, .95, 1]).tolist())
print("Elapsed seconds:", round(time.time()-t84, 1), "edits:", edited())

with open("/work/notes.md", "a") as f84:
    f84.write("\n## Python 84; forward 6 — full posterior reconstruction\n"
              f"- Atom-count mismatches: {int(np.sum(counts84 != S83['n_atoms']))}.\n"
              f"- Best clean conventions: {clean_results84[:1]}.\n"
              f"- Best END conventions: {end_results84[:1]}.\n"
              f"- Full posterior RMS: {best_rms84:.8g}; no weights changed.\n")
```
