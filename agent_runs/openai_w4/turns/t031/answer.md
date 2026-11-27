Independent training states are now available. I will align them to readings, fit an insulin dose-innovation decoder while suppressing MAP changes, and check it on the untouched tuning states. Reliable transfer and a small estimated likelihood cost would justify a conservative head edit. Poor transfer or substantial cross-effects would reject this route.

```python
state_train_repair = dict(np.load("/work/fwd_5/states_original.npz"))
_measure_events = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
_code_map_frozen = {
    ev: int(fact_code_repair[np.flatnonzero(baseline_fact["event"].to_numpy() == ev)[0]])
    for ev in _measure_events
}
_centres_map_frozen = {
    ev: fact_centres_repair[np.flatnonzero(baseline_fact["event"].to_numpy() == ev)[0]].copy()
    for ev in _measure_events
}

# Infer the exact edge convention from the already aligned factual arrays.
_bin_rules = {}
for ev in _measure_events:
    info = numeric_lookup_repair[(ev, "value")]
    edges = np.asarray(info["fine_bin_edges"], dtype=float)
    mask = baseline_fact["event"].to_numpy() == ev
    vals = baseline_fact.loc[mask, "observed"].to_numpy()
    truth_bins = fact_observed_bin_repair[mask]
    trials = []
    for side in ["left", "right"]:
        for offset in [0, 1]:
            pred = np.clip(np.searchsorted(edges, vals, side=side) + offset, 1, 16)
            trials.append((np.mean(pred != truth_bins), side, offset))
    _bin_rules[ev] = min(trials)
print("bin-rule mismatch:", {ev: v for ev, v in _bin_rules.items()})

def reading_bin_fixed(ev, value):
    _, side, offset = _bin_rules[ev]
    edges = np.asarray(numeric_lookup_repair[(ev, "value")]["fine_bin_edges"])
    return int(np.clip(np.searchsorted(edges, value, side=side) + offset, 1, 16))

def build_head_dataset_fixed(states, records, reference=None):
    ids = np.unique(states["stay_id"])
    rec = records[records["stay_id"].isin(ids)].copy()
    if reference is None:
        rr = rec[rec["event"].isin(_measure_events)][
            ["stay_id", "hour", "event", "value"]].rename(columns={"value": "observed"})
        rr = rr.sort_values(["stay_id", "hour", "event"]).reset_index(drop=True)
    else:
        rr = reference[["stay_id", "hour", "event", "observed"]].reset_index(drop=True).copy()
    state_banks = {}
    for sid in ids:
        ii = np.flatnonzero(states["stay_id"] == sid)
        order = np.argsort(states["hour"][ii], kind="stable")
        state_banks[int(sid)] = (states["hour"][ii[order]], ii[order])
    reading_banks = {}
    for (sid, ev), g in rec[rec["event"].isin(_measure_events)].groupby(["stay_id", "event"]):
        g = g.sort_values("hour")
        reading_banks[(int(sid), ev)] = (g["hour"].to_numpy(), g["value"].to_numpy())
    rows = []
    for r in rr.itertuples(index=False):
        sid, hour, ev = int(r.stay_id), float(r.hour), r.event
        sh, si = state_banks[sid]
        k = np.searchsorted(sh, hour, side="left") - 1
        if k < 0:
            continue
        rh, rv = reading_banks[(sid, ev)]
        j = np.searchsorted(rh, hour, side="left") - 1
        if j >= 0:
            info = numeric_lookup_repair[(ev, "value")]
            last_z = (rv[j] - float(info["median"])) / float(info["iqr"])
            age = np.log1p((hour-rh[j])*3600) / 10
            has, last_bin = 1.0, reading_bin_fixed(ev, rv[j])
        else:
            last_z, age, has, last_bin = 0.0, 0.0, 0.0, 0
        rows.append((sid, hour, ev, float(r.observed), int(si[k]),
                     (hour-sh[k])*3600, last_z, age, has, last_bin,
                     _code_map_frozen[ev], reading_bin_fixed(ev, r.observed)))
    return pd.DataFrame(rows, columns=["stay_id", "hour", "event", "observed",
        "prev", "dt", "last_z", "age", "has", "last_bin", "code", "observed_bin"])

def head_features_fixed(states, dataset, draws=16, seed=2501):
    sx = torch.as_tensor(np.asarray(states["x"])[dataset["prev"].to_numpy()],
                         dtype=torch.float32)
    dt = torch.as_tensor(dataset["dt"].to_numpy(), dtype=torch.float32)
    hf = sx[:, 192:960].reshape(-1, 4, 192) * torch.exp(
        -dt[:, None, None] * torch.exp(head_W_repair["core.flow.a"]))
    ctx = sx[:, :192] + head_norm_fixed(hf.mean(1), "core.out_norm") \
          + head_mlp_fixed(torch.log1p(dt)[:, None]/10, "gap")
    prior = head_mlp_fixed(ctx, "z_prior")
    lf = torch.zeros((len(dataset), 8))
    for col, field in [(0, "has"), (3, "age"), (6, "last_z")]:
        lf[:, col] = torch.as_tensor(dataset[field].to_numpy(), dtype=torch.float32)
    codes = torch.as_tensor(dataset["code"].to_numpy(), dtype=torch.long)
    bins = torch.as_tensor(dataset["last_bin"].to_numpy(), dtype=torch.long)
    reg = head_W_repair["embed.code.weight"][codes] + head_linear_fixed(lf, "last_numeric") \
          + head_W_repair["embed.bins.value.weight"][bins] \
          + head_W_repair["embed.bins.rate.weight"][0]
    trans = head_W_repair["embed_transition.value_bin.weight"][codes*17+bins]
    gen = torch.Generator().manual_seed(seed)
    phis, probs = torch.zeros((len(dataset), 192)), torch.zeros((len(dataset), 17))
    for j in range(draws):
        zp = prior[:, :32] + torch.exp(prior[:, 32:].clamp(-10, 5)) * \
             torch.randn((len(dataset), 32), generator=gen)
        q = head_norm_fixed(ctx + head_linear_fixed(zp, "z_head")
                            + head_W_repair["type_emb.weight"][1], "query_norm")
        phi = F.gelu(head_linear_fixed(q+reg, "attr_heads.value_bin.0"))
        phis += phi
        probs += torch.softmax(head_linear_fixed(phi, "attr_heads.value_bin.2")+trans, 1)
    return (phis/draws).numpy(), (probs/draws).numpy()

_tcheck = build_head_dataset_fixed(state_o, tuning, baseline_fact)
print("tuning alignment rows:", len(_tcheck))
assert len(_tcheck) == len(baseline_fact)
print("alignment max errors:", {
    "prev": int(np.max(np.abs(_tcheck["prev"].to_numpy()-fact_prev_repair))),
    "dt": float(np.max(np.abs(_tcheck["dt"].to_numpy()-fact_dt_repair))),
    "age": float(np.max(np.abs(_tcheck["age"].to_numpy()-fact_age_repair))),
    "last_z": float(np.max(np.abs(_tcheck["last_z"].to_numpy()-fact_last_z_repair))),
    "last_bin": int(np.max(np.abs(_tcheck["last_bin"].to_numpy()-fact_bin_repair)))
})
assert np.array_equal(_tcheck["prev"].to_numpy(), fact_prev_repair)
assert np.max(np.abs(_tcheck["last_z"].to_numpy()-fact_last_z_repair)) < 1e-4
assert np.array_equal(_tcheck["last_bin"].to_numpy(), fact_bin_repair)

head_train_data_repair = build_head_dataset_fixed(state_train_repair, train)
head_train_phi_repair, head_train_prob_repair = head_features_fixed(
    state_train_repair, head_train_data_repair, draws=16)
head_tuning_data_repair = _tcheck

# Recover which nuisance model was held out for each original DML fold.
_fold_to_model = {}
_res = dml_residuals["insulin"]
for f in np.unique(fold):
    q = np.flatnonzero(fold[_res["indices"]] == f)[:100]
    ii = _res["indices"][q]
    errors = [np.mean(np.abs(m[0].predict(HX.iloc[ii].to_numpy())
                                - _res["dose_prediction"][q]))
              for m in dml_models["insulin"]]
    _fold_to_model[int(f)] = int(np.argmin(errors))
print("cross-fitting map:", _fold_to_model)

def insulin_labels_fixed(dataset, panel, hist, is_train):
    pos = panel.index.get_indexer(pd.MultiIndex.from_arrays([
        dataset["stay_id"].to_numpy(),
        np.rint(dataset["hour"].to_numpy()).astype(int)-1]))
    good = (pos >= 0) & (dataset["hour"].to_numpy() >= 1)
    eps = np.zeros(len(dataset))
    jj = np.flatnonzero(good)
    pp = pos[jj]
    if is_train:
        pred = np.zeros(len(jj))
        for f, mi in _fold_to_model.items():
            sel = fold[pp] == f
            pred[sel] = dml_models["insulin"][mi][0].predict(hist.iloc[pp[sel]].to_numpy())
    else:
        pred = np.asarray(policy_pred_tuning["insulin"])[pp]
    eps[jj] = panel["insulin"].fillna(0).to_numpy()[pp] - pred
    glu = dataset["event"].isin(["glucose_finger", "glucose_lab"]).to_numpy()
    target = np.where(good & glu, eps, 0.0)
    return target, eps, good, glu

_ytr, _etr, _gtr, _gltr = insulin_labels_fixed(
    head_train_data_repair, pan, HX, True)
_yva, _eva, _gva, _glva = insulin_labels_fixed(
    head_tuning_data_repair, tpan, tHX_repair, False)
_ktr = (head_train_data_repair["code"]*17 + head_train_data_repair["last_bin"]).to_numpy()
_kva = (head_tuning_data_repair["code"]*17 + head_tuning_data_repair["last_bin"]).to_numpy()
_group_means = np.zeros((221, 192))
for ev in _measure_events:
    c = _code_map_frozen[ev]
    mm = head_train_data_repair["event"].to_numpy() == ev
    _group_means[c*17:(c+1)*17] = head_train_phi_repair[mm].mean(0)
for k in np.unique(_ktr):
    mm = _ktr == k
    if mm.sum() >= 5:
        _group_means[k] = head_train_phi_repair[mm].mean(0)
_Ftr = head_train_phi_repair - _group_means[_ktr]
_Fva = head_phi_repair - _group_means[_kva]
_scale_decode = np.maximum(_Ftr.std(0), .03)
decoder_candidates_repair = {}
print("\n=== INDEPENDENT TRAIN-TO-TUNING DECODERS ===")
for alpha in [10, 100, 1000, 10000]:
    reg = Ridge(alpha=alpha, fit_intercept=False).fit(
        _Ftr/_scale_decode, _ytr)
    w = reg.coef_/_scale_decode
    pv = _Fva @ w
    mm = _gva & _glva
    r2 = r2_score(_eva[mm], pv[mm])
    covariance_fraction = np.mean(_eva[mm]*pv[mm])/np.mean(_eva[mm]**2)
    map_sd = float(pv[~_glva].std())
    print("alpha", alpha, "glucose innovation R2", round(float(r2), 4),
          "covariance fraction", round(float(covariance_fraction), 4),
          "MAP spurious-score SD", round(map_sd, 4))
    decoder_candidates_repair[alpha] = dict(
        coef=w, prediction=pv, r2=r2, covariance_fraction=covariance_fraction,
        map_sd=map_sd)
_best_alpha = max(decoder_candidates_repair,
                  key=lambda a: decoder_candidates_repair[a]["r2"])
insulin_decoder_repair = dict(decoder_candidates_repair[_best_alpha],
    group_means=_group_means, alpha=_best_alpha)
print("selected statistical-model alpha:", _best_alpha)

# Approximate factual cost of a bin-logit tilt; no weights are changed here.
_gc = _centres_map_frozen["glucose_finger"]
_ginfo = numeric_lookup_repair[("glucose_finger", "value")]
tilt_scores_repair = (_gc-float(_ginfo["median"]))/float(_ginfo["iqr"])
tilt_scores_repair[0] = 0.0
_mu = (head_prob_repair*fact_centres_repair).sum(1)
_kappa = (head_prob_repair*(fact_centres_repair-_mu[:, None])
          *tilt_scores_repair[None, :]).sum(1)
_kappa_glu = float(np.mean(_kappa[_glva & _gva]))
insulin_logit_unit_repair = -2.04924/_kappa_glu
print("glucose mean-tilt derivative:", round(_kappa_glu, 4))
print("\n=== APPROXIMATE FACTUAL COST, NOT A FORWARD-TOOL RESULT ===")
for gain in [1, 2, 3, 4]:
    s = gain*insulin_logit_unit_repair*insulin_decoder_repair["prediction"]
    pp = softmax(np.log(np.maximum(head_prob_repair, 1e-20))
                 + s[:, None]*tilt_scores_repair, axis=1)
    dnll = -np.log(np.maximum(pp[np.arange(len(pp)), fact_observed_bin_repair], 1e-20)) \
           + np.log(np.maximum(head_prob_repair[
               np.arange(len(pp)), fact_observed_bin_repair], 1e-20))
    print("gain", gain, "overall delta NLL", round(float(dnll.mean()), 6),
          "by event", pd.Series(dnll).groupby(
              head_tuning_data_repair["event"]).mean().round(6).to_dict())

print("Current edits:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 25; forward passes used: 4\n"
            "Built reusable head dataset and feature functions, verified tuning alignment, "
            "fit insulin-innovation decoders on independent training states with MAP scores "
            "penalised toward zero, and checked tuning transfer plus approximate factual costs. "
            "No weights edited. Selected decoder alpha: " + str(_best_alpha) + ".\n")
```