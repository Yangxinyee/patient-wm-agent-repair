The coda passes the private signals almost unchanged, with little leakage into other state directions. I’ll now fit an analytical extractor of dose changes from its inputs, penalizing any signal on factual records. Accurate contrasts across stopping, halving, and increasing doses—with small factual residuals—would justify trying a weight edit.

```python
# Fit small linear extractors; no model gradients or weight optimization.
base_ids123 = np.asarray(valid_indices103, dtype=np.int64)
assert len(base_ids123) == len(Hpre118)
assert np.array_equal(base_ids123, valid116)
lookup123 = np.full(len(sx63), -1, dtype=np.int64)
lookup123[base_ids123] = np.arange(len(base_ids123))
pre_cf123 = {}
capture123 = []
linear_before123 = linear63

def capture_pre123(x, prefix):
    if prefix == "core.c_out":
        capture123.append(x.detach().cpu().numpy().copy())
    return linear_backup118(x, prefix)

try:
    linear63 = capture_pre123
    for drug123 in ("insulin", "norepinephrine"):
        drug_mask123 = (drug_names81 == drug123)
        selected123 = base_ids123[
            np.asarray(rate_state81[drug123])[base_ids123] > 0]
        for factor123 in (0.0, 0.5, 1.5, 2.0, 3.0):
            rr123 = rates65.copy()
            dd123 = doses65.copy()
            rr123[drug_mask123] *= factor123
            dd123[drug_mask123] *= factor123
            uc123 = pool_actions49(
                action_embedding67(drugs65, rr123, dd123),
                owners72, counts72)
            capture123.clear()
            for start123 in range(0, len(selected123), 96):
                ids123 = selected123[start123:start123+96]
                ctx123, hf123, mem123, age123, mask123 = bundle_inputs74(ids123)
                zz123 = mlp63(torch.cat(
                    [ctx123, e_valid60[ids123], uc123[ids123]], -1),
                    "z_post")[:, :32]
                transition64(hf123, e_valid60[ids123], uc123[ids123],
                             zz123, mem123, age123, mask123)
            pre_cf123[(drug123, factor123)] = {
                "ids": selected123.copy(),
                "features": np.concatenate(capture123).mean(1),
                "delta_rate": ((factor123-1) *
                               np.asarray(rate_state81[drug123])[selected123])
            }
finally:
    linear63 = linear_before123
    capture123.clear()

F123 = np.c_[Hpre118.reshape(-1, 192).astype(np.float64),
             np.ones(len(Hpre118)*4)]
Dparts123, Yparts123, group123 = [], [], []
for (drug123, factor123), item123 in pre_cf123.items():
    ids123 = item123["ids"]
    dx123 = (item123["features"].astype(np.float64)
             - Hpre118[lookup123[ids123]].mean(1))
    yy123 = np.zeros((len(ids123), 2))
    yy123[:, int(drug123 == "norepinephrine")] = item123["delta_rate"]
    Dparts123.append(np.c_[dx123, np.zeros(len(ids123))])
    Yparts123.append(yy123)
    group123.extend([(drug123, factor123)]*len(ids123))
D123, Y123 = np.concatenate(Dparts123), np.concatenate(Yparts123)
Sf123 = F123.T @ F123 / len(F123)
Sc123 = D123.T @ D123 / len(D123)
rhs123 = D123.T @ Y123 / len(D123)
ridge123 = np.eye(193) * max(np.trace(Sc123)/193, 1e-8)*1e-7
fits123, summaries123 = {}, {}
for penalty123 in (0.01, 0.1, 1.0, 10.0):
    B123 = np.linalg.solve(Sc123 + penalty123*Sf123 + ridge123, rhs123)
    fits123[penalty123] = B123
    pred123 = D123 @ B123
    factual123 = F123 @ B123
    slopes123 = {}
    offset123 = 0
    for key123, item123 in pre_cf123.items():
        nn123 = len(item123["ids"])
        col123 = int(key123[0] == "norepinephrine")
        y123 = item123["delta_rate"]
        p123 = pred123[offset123:offset123+nn123, col123]
        slopes123[key123] = round(float(p123@y123 / (y123@y123)), 3)
        offset123 += nn123
    summaries123[penalty123] = {
        "factual_RMS": np.sqrt(np.mean(factual123**2, axis=0)).round(4).tolist(),
        "contrast_RMSE": np.sqrt(np.mean((pred123-Y123)**2, axis=0)).round(4).tolist(),
        "coefficient_norms": np.linalg.norm(B123[:-1], axis=0).round(2).tolist(),
        "slopes": slopes123
    }
print("Pre-coda dose-extractor results:")
for k123, v123 in summaries123.items():
    print(k123, v123)
print("Current edits:", edited())

with open("/work/notes.md", "a") as f123:
    f123.write("\n## Run 123 / pre-coda dose extraction\n"
               "Private directions transmit through coda with gains≈0.998 "
               "and orthogonal RMS leakage≈0.001–0.003 per unit. "
               "Fit covariance-penalized linear dose extractors from factual "
               "and counterfactual pre-coda states, including x0/x0.5/x1.5/x2/x3. "
               + str(summaries123) + "\nNo weights edited.\n")
```