The best reconstruction is close, but not yet accurate enough: slot RMSE is about 1.03 and output RMSE about 0.015. It points to memory including the current bundle, with age added before both key and value projections. I’ll localize the remaining error by bundle type; errors confined to extra sensors or visits would indicate an encoding issue rather than a transition issue.

```python
import time
t125 = time.time()

rec_world_valid = np.flatnonzero(
    (rec_counts.numpy() > 0)
    & ~rm_states["has_insulin"] & ~rm_states["has_norepinephrine"])
rec_world_valid = rec_world_valid[
    (rec_world_valid > 0)
    & (rm_states["stay_id"][rec_world_valid]
       == rm_states["stay_id"][np.maximum(rec_world_valid - 1, 0)])]
rec_wdt = ((rm_states["hour"][rec_world_valid]
            - rm_states["hour"][rec_world_valid - 1]) * 3600)[:, None]
rec_wctx, _ = rec_context(rm_states["x"][rec_world_valid - 1], rec_wdt)
rec_wu = empty_set_vector("encode_action", len(rec_world_valid))
rec_wz = rm_mlp(torch.cat([rec_wctx, rec_e_all[rec_world_valid], rec_wu], -1),
               rec_zpost)[:, :32]
rec_wtarget = torch.as_tensor(rm_states["x"][rec_world_valid, 960:])
rec_werror = (rec_wz - rec_wtarget).square().mean(1).sqrt().numpy()

rec_compositions = {}
for key in rm_lookup:
    if key[2] not in ("insulin", "norepinephrine"):
        rec_compositions.setdefault(key[:2], []).append(key[2])
rec_wlabels = [",".join(sorted(rec_compositions.get(
    (int(rm_states["stay_id"][ix]), float(rm_states["hour"][ix])), [])))
    for ix in rec_world_valid]
rec_world_diag = pd.DataFrame({"ix": rec_world_valid, "composition": rec_wlabels,
                               "posterior_rmse": rec_werror})
print("POSTERIOR ERRORS BY WORLD-BUNDLE COMPOSITION:")
print(rec_world_diag.groupby("composition").posterior_rmse.agg(
    ["size", "median", "mean", "max"]).to_string())

def rec_memory(indices, embeddings=None):
    embeddings = rec_e_all if embeddings is None else embeddings
    selections = []
    for ix in indices:
        js = np.flatnonzero(
            (rm_states["stay_id"] == rm_states["stay_id"][ix])
            & (np.arange(len(rm_states["x"])) <= ix))[-128:]
        selections.append(js)
    width = max(map(len, selections))
    mem = torch.zeros((len(indices), width, 192))
    ages = torch.zeros((len(indices), width, 1))
    mask = torch.zeros((len(indices), width), dtype=torch.bool)
    for j, js in enumerate(selections):
        mem[j, :len(js)] = embeddings[js]
        ages[j, :len(js), 0] = torch.tensor(
            (rm_states["hour"][indices[j]] - rm_states["hour"][js]) * 3600,
            dtype=torch.float32)
        mask[j, :len(js)] = True
    return mem, rm_mlp(torch.log1p(ages), "core.age"), mask

def rec_step_with_z(prev, dt, e_world, u_action, z, memory, ages, mask):
    hf = rec_context(prev, dt)[1]
    tokens = torch.cat([hf + rm_W["core.slots"][None], e_world[:, None],
                        u_action[:, None], rm_lin(z, rec_zin)[:, None]], 1)
    E = rm_ln(rec_layer(tokens, "core.prelude.0")[:, :4], "core.prelude_norm")
    hbase = hf + E
    u = hbase
    for j in range(2):
        u = rec_layer(u, "core.blocks.0." + str(j), memory, ages, mask, "both_input")
    hn = rec_abar * hf + rec_delta * rm_lin(E, "core.b_in") + u - hbase
    hn = rec_layer(rm_lin(hn, "core.c_out"), "core.coda.0")
    return hn, rm_ln(hn.mean(1), rec_outnorm)

rec_mem48, rec_age48, rec_mask48 = rec_memory(rm_pure_idx)
rec_h48, rec_o48 = rec_step_with_z(
    rm_prev, rm_dt, r96_best["e"], r96_best["u"], rm_target_z,
    rec_mem48, rec_age48, rec_mask48)
rec_transition_diag = pd.DataFrame({
    "stay_id": rm_states["stay_id"][rm_pure_idx],
    "hour": rm_states["hour"][rm_pure_idx],
    "slot_rmse": (rec_h48 - rec_h_target).square().mean((1, 2)).sqrt().numpy(),
    "out_rmse": (rec_o48 - rec_out_target).square().mean(1).sqrt().numpy()})
print("\nTRANSITION ERRORS:")
print(rec_transition_diag.head(12).to_string(index=False))
print("SECONDS:", round(time.time() - t125, 2))
```