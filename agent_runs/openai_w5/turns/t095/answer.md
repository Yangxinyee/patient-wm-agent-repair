The local recurrence still differs from the saved GPU state, and its corrected latent response remains negligible. I’ll stop pursuing latent edits and test whether the prelude’s direct action features retain a usable dose signal. A useful decoder must generalize to other stays and predict finite dose changes without producing large changes on factual bundles.

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

wm_kv_layout = "global"
wm_memory_mode = "unnormalized"
step_options83 = dict(include_current=True, age_mode="input",
                      age_units=3600.0, age_scale=1.0)

def prelude_all83(action_u, rows=rows4):
    rows = np.asarray(rows, dtype=int)
    hh = torch.as_tensor(s4["x"][rows-1, 192:960]).reshape(-1, 4, 192)
    dt = np.round((s4["hour"][rows] - s4["hour"][rows-1])*60)*60
    hh = hh * torch.exp(-torch.as_tensor(dt, dtype=torch.float32)[:, None, None]
                        * torch.exp(W0["core.flow.a"]))
    ee = wm_e4[rows]
    uu = torch.as_tensor(action_u)[rows]
    zz = torch.as_tensor(s4["x"][rows, 960:992])
    tokens = torch.cat([hh + W0["core.slots"], ee[:, None],
                        uu[:, None], wm_lin(zz, "z_in")[:, None]], dim=1)
    pre = wm_layer(tokens, "core.prelude.0")
    return wm_norm(pre[:, :4], "core.prelude_norm").mean(1).numpy()

rates83 = np.zeros((len(s4["hour"]), 2), dtype=np.float64)
for d83, drug83 in enumerate(("insulin", "norepinephrine")):
    tab83 = acttable4[acttable4.event == drug83]
    np.add.at(rates83[:, d83], tab83.j.to_numpy(dtype=int), tab83.rate.to_numpy())
active83 = np.column_stack([s4["has_insulin"], s4["has_norepinephrine"]])[rows4]
innov83 = np.asarray(innovation55, dtype=np.float64).copy()
innov83[~active83] = 0
E83 = prelude_all83(U4)
print("Factual E shape / RMS:", E83.shape, np.sqrt(np.mean(E83**2)))
print("Dose / innovation standard deviations:",
      rates83[rows4].std(0), innov83.std(0))

cf83 = {}
for drug83 in ("insulin", "norepinephrine"):
    for factor83 in (0., .5, 1.5, 2., 3.):
        cf83[(drug83, factor83)] = prelude_all83(action_vectors4(drug83, factor83)) - E83
print("Action-change feature RMS:",
      {str(k): round(float(np.sqrt(np.mean(v*v))), 7) for k, v in cf83.items()})

ids83 = np.unique(s4["stay_id"])
fit_ids83 = ids83[:60]
fit83 = np.isin(s4["stay_id"][rows4], fit_ids83)
scale83 = np.maximum(E83[fit83].std(0), 1e-5)
Xfact83 = np.column_stack([E83 / scale83, np.ones(len(E83))])
DX83, DY83, DM83 = [], [], []
for (drug83, factor83), deltaE83 in cf83.items():
    d83 = ("insulin", "norepinephrine").index(drug83)
    a83 = active83[:, d83]
    y83 = np.zeros((a83.sum(), 2))
    y83[:, d83] = rates83[rows4[a83], d83] * (factor83 - 1)
    DX83.append(np.column_stack([deltaE83[a83] / scale83, np.zeros(a83.sum())]))
    DY83.append(y83)
    DM83.append(fit83[a83])
DX83, DY83, DM83 = np.vstack(DX83), np.vstack(DY83), np.concatenate(DM83)

fits83, report83 = {}, []
for target83_name, factual_y83 in (("innovation", innov83),
                                   ("preserve", np.zeros_like(innov83))):
    for weight83 in (1., 100., 10000.):
        xx83 = np.vstack([Xfact83[fit83], DX83[DM83]*np.sqrt(weight83)])
        yy83 = np.vstack([factual_y83[fit83], DY83[DM83]*np.sqrt(weight83)])
        fit_obj83 = Ridge(alpha=.01, fit_intercept=False).fit(xx83, yy83)
        fits83[(target83_name, weight83)] = fit_obj83
        p83 = fit_obj83.predict(Xfact83[~fit83])
        dp83 = fit_obj83.predict(DX83[~DM83])
        report83.append({
            "target": target83_name, "cf_weight": weight83,
            "factual_sd_I": p83[:, 0].std(), "factual_sd_NE": p83[:, 1].std(),
            "innovation_R2_I": r2_score(innov83[~fit83, 0], p83[:, 0]),
            "innovation_R2_NE": r2_score(innov83[~fit83, 1], p83[:, 1]),
            "cf_R2_I": r2_score(DY83[~DM83, 0], dp83[:, 0]),
            "cf_R2_NE": r2_score(DY83[~DM83, 1], dp83[:, 1]),
            "coef_norm": np.linalg.norm(fit_obj83.coef_)})
print(pd.DataFrame(report83).round(4).to_string(index=False))
print("Current edits:", edited())
```