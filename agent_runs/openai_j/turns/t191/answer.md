The stronger available checks agree that norepinephrine increases MAP, although their magnitude and persistence estimates differ. I’ll treat +3.8/−1.9 as an uncertain target, not known truth. Next I’ll evaluate MAP-only corrections jointly with the working insulin edit; only candidates with factual-fit headroom will proceed to rollout testing.

```python
map172_glucose_train = np.isin(cal170_train_events, ["glucose_finger", "glucose_lab"])
map172_glucose_tune = np.isin(cal170_tune_events, ["glucose_finger", "glucose_lab"])
map172_MAP_train = ~map172_glucose_train
map172_MAP_tune = ~map172_glucose_tune

def map172_tau(mc, insulin_coef, map_coef, glucose_mask):
    x = mc["inputs"]
    iw = torch.as_tensor(insulin_coef, dtype=x.dtype)
    mw = torch.as_tensor(map_coef, dtype=x.dtype)
    gm = torch.as_tensor(glucose_mask, dtype=x.dtype)
    return ((x @ iw[:192] + iw[192]) * gm
            + (x @ mw[:192] + mw[192]) * (1 - gm))

def map172_mean(mc, tau, centres):
    return (torch.softmax(mc["logits"] + tau[..., None] * wm_v151, -1)
            .mean(0) * centres).sum(-1).numpy()

map172_rows = []
map172_candidates = {}
for fw172 in [0.3, 1.0, 3.0]:
    for gain172 in [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]:
        coef172 = gain172 * map164_fits[(0.0, fw172, 0.001)]
        map172_candidates[(fw172, gain172)] = coef172
        tau172 = map172_tau(wm_train145, trial158_coef, coef172, map172_glucose_train)
        ttau172 = map172_tau(wm_tune145, trial158_coef, coef172, map172_glucose_tune)
        dn172 = (cal170_nll(wm_train145, tau172[..., None] * wm_v151).numpy()
                 - cal170_base_train)
        tdn172 = (cal170_nll(wm_tune145, ttau172[..., None] * wm_v151).numpy()
                  - cal170_base_tune)
        base_mu172 = map172_mean(wm_train145, tau172, wm_train145["centres"])
        row172 = {"fw": fw172, "gain": gain172,
                  "heldout_dNLL": dn172[~wm_split151].mean(),
                  "tuning_dNLL": tdn172.mean()}
        for factor172 in [0.5, 2.0]:
            cf172 = wm_cf146[("norepinephrine", factor172)]
            ctau172 = map172_tau(cf172, trial158_coef, coef172, map172_glucose_train)
            cmu172 = map172_mean(cf172, ctau172, wm_train145["centres"])
            sel172 = cf172["active"] & (cal170_train_events == "map_arterial")
            dr172 = cf172["delta_rate"][sel172]
            dy172 = cmu172[sel172] - base_mu172[sel172]
            row172[f"N{factor172}_slope"] = np.dot(dr172, dy172) / np.dot(dr172, dr172)
            row172[f"N{factor172}_shift"] = dy172.mean()
        map172_rows.append(row172)
map172_frontier = pd.DataFrame(map172_rows)
print(map172_frontier.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
print("\nExisting insulin units:", trial158_free)
print("Saved fallback types:", {k: type(v).__name__ for k, v in best_insulin_v1.items()})
```
