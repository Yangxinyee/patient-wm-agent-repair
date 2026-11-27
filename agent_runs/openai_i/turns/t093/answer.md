Bundle construction now matches every exposed atom count. The numerical test stopped because the retained helper expected an activation function, not the string `"gelu"`; no weights changed. I’ll fix that interface and rerun the check. Posterior errors near floating-point precision would confirm the reconstruction.

```python
t85 = time.time()
old_mlp85 = repair_mlp_act
def repair_mlp_act(x, name, activation):
    if isinstance(activation, str):
        activation = {"gelu": torch.nn.functional.gelu,
                      "silu": torch.nn.functional.silu}[activation]
    return repair_linear(activation(repair_linear(x, name+".0")), name+".2")

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
    return f, repair_linear(f, "z_post.2")[:, :32]

clean84 = torch.from_numpy(valid84 & (endcount84 == 0))
clean_results84 = []
for rd85, lp85, ee85, eu85 in itertools.product(
        [True, False], [False, True], ["encoded", "zero"], ["encoded", "zero"]):
    c85 = context84(rd85, lp85)
    e85, u85 = Eworld84.clone(), Ubase84.clone()
    if ee85 == "zero": e85[obscount84 == 0] = 0
    if eu85 == "zero": u85[actioncount84 == 0] = 0
    _, z85 = posterior84(c85, e85, u85)
    rms85 = float(((z85[clean84]-Ztarget84[clean84])**2).mean().sqrt())
    clean_results84.append((rms85, rd85, lp85, ee85, eu85))
clean_results84.sort()
print("Best clean conventions:", clean_results84[:4])
_, rounded84, log1p84, empty_e84, empty_u84 = clean_results84[0]
C84 = context84(rounded84, log1p84)
U84 = Ubase84.clone()
if empty_u84 == "zero": U84[actioncount84 == 0] = 0

world_sum85 = torch.zeros((nb84, 192))
world_sum85.index_add_(0, torch.tensor(world_indices84),
                      repair_mlp_act(torch.stack(world_atoms84), "encode.mlp1", "gelu"))
end_base85 = {}
for numbers85 in [False, True]:
    end_base85[numbers85] = torch.stack([
        complete_atom57(r, kind=0, ekind=0, unit=0, rate_unit=0, numbers=numbers85)
        for r in end_rows84])
end_unit85 = torch.tensor([unit84.get(r["event"], 0) for r in end_rows84])
end_rate85 = torch.tensor([rateunit84.get(r["event"], 0) for r in end_rows84])
end_ekind85 = torch.tensor([ekind_ids57[event_info57[r["event"]]["event_kind"]]
                            for r in end_rows84])
end_index85 = torch.tensor(end_indices84)
obsn85 = torch.tensor(obscount84, dtype=torch.float32)[:, None]
obs_size85 = repair_linear(torch.log1p(obsn85), "encode.size")
mask84 = torch.from_numpy(valid84)
end_results84, best_rms84 = [], np.inf

for eu85, er85, en85, ek85, ee85 in itertools.product(
        ["absent", "same", "one"], ["absent", "same", "one"],
        [False, True], [2, 0, 1], ["same", "zero"]):
    ui85 = end_unit85 if eu85 == "same" else torch.full_like(end_unit85, int(eu85=="one"))
    ri85 = end_rate85 if er85 == "same" else torch.full_like(end_rate85, int(er85=="one"))
    ei85 = end_ekind85 if ee85 == "same" else torch.zeros_like(end_ekind85)
    av85 = (end_base85[en85]
            + repair_weights["embed.kind.weight"][ek85] - repair_weights["embed.kind.weight"][0]
            + repair_weights["embed.ekind.weight"][ei85] - repair_weights["embed.ekind.weight"][0]
            + repair_weights["embed.categorical.unit.weight"][ui85] - repair_weights["embed.categorical.unit.weight"][0]
            + repair_weights["embed.categorical.rate_unit.weight"][ri85] - repair_weights["embed.categorical.rate_unit.weight"][0])
    sums85 = world_sum85.clone()
    sums85.index_add_(0, end_index85, repair_mlp_act(av85, "encode.mlp1", "gelu"))
    e85 = repair_norm(repair_mlp_act(
        sums85/obsn85.clamp_min(1).sqrt()+obs_size85, "encode.mlp2", "gelu"), "encode.norm")
    if empty_e84 == "zero": e85[obscount84 == 0] = 0
    f85, z85 = posterior84(C84, e85, U84)
    rms85 = float(((z85[mask84]-Ztarget84[mask84])**2).mean().sqrt())
    cfg85 = (eu85, er85, en85, ek85, ee85)
    end_results84.append((rms85, cfg85))
    if rms85 < best_rms84:
        best_rms84 = rms85
        E84, F84, Z84, end_config84 = e85.clone(), f85.clone(), z85.clone(), cfg85

end_results84.sort()
print("Best END conventions:", end_results84[:5])
err85 = ((Z84-Ztarget84)**2).mean(1).sqrt().numpy()
for label85, m85 in [
    ("all", valid84), ("actions", valid84 & (actioncount84 > 0)),
    ("no END/no action", valid84 & (actioncount84 == 0) & (endcount84 == 0)),
    ("END present", valid84 & (endcount84 > 0))]:
    print(label85, "n", int(m85.sum()),
          "median/p95/max RMS", np.quantile(err85[m85], [.5, .95, 1]).tolist())
print("Elapsed:", round(time.time()-t85, 1), "seconds; edits:", edited())
with open("/work/notes.md", "a") as f85:
    f85.write("\n## Python 85; forward 6 — posterior reconstruction check\n"
              "- Python 84 established zero atom-count mismatches but stopped on an activation-helper interface error.\n"
              f"- Clean conventions: {clean_results84[:1]}; END conventions: {end_results84[:1]}.\n"
              f"- Full posterior RMS {best_rms84:.8g}; no weights changed.\n")
```
