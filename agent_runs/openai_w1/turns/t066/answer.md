Four posterior neurons are effectively inactive, so they may be reusable with negligible factual disruption. The norepinephrine offset also extrapolates to impossible rates on some bundles, so it needs bounds. I’ll first verify the dose traces and estimate latent reading directions. Accurate held-out traces and stable directions would support a small trial.

```python
from scipy.linalg import solve

rr58_trace_coef = rr51_trace_candidates[rr51_trace_alpha][1]
rr58_trace_bias = rr51_trace_candidates[rr51_trace_alpha][2]
rr58_P = rr50_P.astype(np.float64)
rr58_projection = np.eye(192) - rr58_P @ rr58_P.T
rr58_Waction = (rr58_projection @ rr50_Wold +
                rr50_epsilon * rr58_P @ rr58_trace_coef).astype(np.float32)
rr58_baction = (rr58_projection @ rr50_bold +
                rr50_epsilon * rr58_P @ rr58_trace_bias).astype(np.float32)
rr58_gamma = np.asarray(load_original("encode_action.norm.weight"), dtype=np.float64)
rr58_normbias = np.asarray(load_original("encode_action.norm.bias"), dtype=np.float64)
rr58_rawread = rr58_P / rr58_gamma[:, None] / rr50_epsilon
rr58_new_u, rr58_trace_features, rr58_targets = {}, {}, {}

for kk, cache in rr50_cache.items():
    hh0, pre0, active0 = cache
    pre1 = np.asarray(hh0) @ rr58_Waction.T + rr58_baction
    u1 = F.layer_norm(torch.tensor(pre1), (192,),
                      torch.tensor(rr58_gamma, dtype=torch.float32),
                      torch.tensor(rr58_normbias, dtype=torch.float32)).numpy()
    u1[np.asarray(active0) == 0] = 0
    rr58_new_u[kk] = u1
    rr58_trace_features[kk] = u1.astype(np.float64) @ rr58_rawread
    rates1 = rr48_rates.copy()
    if isinstance(kk[0], int):
        rates1[:, kk[0]] *= kk[1]
    rr58_targets[kk] = np.column_stack((rates1 / rr50_scales, rr48_present))

rr58_fitF = np.concatenate([v[rr49_train] for v in rr58_trace_features.values()])
rr58_fitT = np.concatenate([v[rr49_train] for v in rr58_targets.values()])
rr58_cal = solve(rr58_fitF.T @ rr58_fitF + 1e-7*np.eye(4),
                 rr58_fitF.T @ rr58_fitT, assume_a="pos")
rr58_read = rr58_rawread @ rr58_cal
print("=== HELD-OUT TRACE ERRORS ===")
for kk in (("base", 1.0), (0, 0.0), (0, 2.0), (0, 3.0), (1, 0.0), (1, 2.0), (1, 3.0)):
    err = rr58_new_u[kk][~rr49_train] @ rr58_read - rr58_targets[kk][~rr49_train]
    active1 = rr48_present[~rr49_train].sum(axis=1) > 0
    print(kk, "RMSE", np.sqrt(np.mean(err[active1]**2, axis=0)).round(5),
          "presence max", round(float(np.max(np.abs(err[active1, 2:]))), 5))
print("factual action-vector RMS change",
      float(np.sqrt(np.mean((rr58_new_u[("base", 1.0)] - np.asarray(rr48_u0))**2))))

def rr58_observation_encoding(bundle_rows, glucose_delta=0.0, map_delta=0.0):
    vectors, owners = [], []
    counts0 = []
    for bi, rowlist in enumerate(bundle_rows):
        count0 = 0
        for row0 in rowlist:
            if row0.event in ("insulin", "norepinephrine"):
                continue
            val0 = row0.value
            if row0.event in ("glucose_finger", "glucose_lab"):
                val0 += glucose_delta
            if row0.event in ("map_arterial", "map_cuff"):
                val0 += map_delta
            vectors.append(r36_atom(row0.event, value=val0))
            owners.append(bi)
            count0 += 1
        counts0.append(count0)
    vectors = torch.stack(vectors)
    atoms0 = mlp(vectors, "encode.mlp1", "gelu")
    pooled = torch.zeros((len(bundle_rows), 192))
    pooled.index_add_(0, torch.tensor(owners), atoms0)
    counts0 = torch.tensor(counts0, dtype=torch.float32)
    pooled = pooled / torch.sqrt(counts0[:, None].clamp_min(1))
    pooled += lin(torch.log1p(counts0[:, None]), "encode.size")
    return norm(mlp(pooled, "encode.mlp2", "gelu"), "encode.norm")

rr58_measurement_ix = []
rr58_measurement_rows = []
for bi in range(len(r36_sid)):
    hour0 = float(r36_hour[bi])
    if 3 <= hour0 <= 23 and abs(hour0-round(hour0)) < 1e-4:
        rows0 = r36_world.get((int(r36_sid[bi]), round(hour0, 4)), [])
        if rows0 and any(row0.event in sensors for row0 in rows0):
            rr58_measurement_ix.append(bi)
            rr58_measurement_rows.append(rows0)
rr58_pick = np.random.default_rng(5801).choice(
    len(rr58_measurement_ix), min(1200, len(rr58_measurement_ix)), replace=False)
rr58_mi = np.asarray(rr58_measurement_ix)[rr58_pick]
rr58_mrows = [rr58_measurement_rows[i] for i in rr58_pick]
rr58_e0 = rr58_observation_encoding(rr58_mrows)
print("measurement encoder reproduction RMSE",
      float(torch.sqrt(torch.mean((rr58_e0-r36_e[rr58_mi])**2))))
rr58_reading_dirs = {}
for family0, step0 in (("glucose", 10.0), ("map", 3.0)):
    kwargs_plus = {"glucose_delta" if family0 == "glucose" else "map_delta": step0}
    kwargs_minus = {"glucose_delta" if family0 == "glucose" else "map_delta": -step0}
    eplus = rr58_observation_encoding(rr58_mrows, **kwargs_plus)
    eminus = rr58_observation_encoding(rr58_mrows, **kwargs_minus)
    inplus = torch.cat((r36_ctx[rr58_mi], eplus, torch.zeros_like(eplus)), dim=1)
    inminus = torch.cat((r36_ctx[rr58_mi], eminus, torch.zeros_like(eminus)), dim=1)
    muplus = mlp(inplus, "z_post", "gelu")[:, :32]
    muminus = mlp(inminus, "z_post", "gelu")[:, :32]
    has_family = np.array([any(r.event.startswith(family0+"_") for r in rows0)
                           for rows0 in rr58_mrows])
    diffs0 = ((muplus-muminus)/(2*step0)).numpy()[has_family]
    rr58_reading_dirs[family0] = np.mean(diffs0, axis=0)
    print(family0, "n", len(diffs0),
          "latent direction norm", float(np.linalg.norm(rr58_reading_dirs[family0])),
          "half-sample cosine", float(
              np.dot(diffs0[::2].mean(0), diffs0[1::2].mean(0)) /
              (np.linalg.norm(diffs0[::2].mean(0))*np.linalg.norm(diffs0[1::2].mean(0))+1e-20)))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 58\n"
            "Built selected trace candidate and its linear reader; checked held-out extrapolation. "
            "Estimated posterior latent reading directions by finite record-value perturbations, no gradients. "
            "No tensors saved. NE expected-dose predictor needs physical bounds.\n"
            "Counts: 58 Python runs, 6 forward passes.\n")
print("CURRENT EDITS:", edited())
```