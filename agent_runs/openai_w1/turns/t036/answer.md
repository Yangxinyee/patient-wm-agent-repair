The enriched sample gives us substantially more treated arterial and glucose readings. I’ll combine its states with the first sample, reconstruct value-head features, and test dose-innovation decoding on held-out stays. If decoding fails out of sample, these features are not a sound basis for the edit.

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

ce_code = {x["name"]: x["id"] for x in VOCAB["codes"]}
ce_meta = {(x["code"], x["attribute"]): x for x in VOCAB["numeric"]}
ce_centres = {
    ev: torch.as_tensor(hcentres)[np.flatnonzero(head_records.event.to_numpy() == ev)[0]].numpy()
    for ev in repair_events
}

def ce_make_heads(records, states):
    rows = records[records.event.isin(repair_events)].copy()
    rows = rows.sort_values(["stay_id", "hour", "event"]).reset_index(drop=True)
    n = len(rows)
    previous = np.zeros((n, 992), np.float32)
    elapsed = np.full(n, 3600.0, np.float32)
    numeric = np.zeros((n, 8), np.float32)
    lastbins = np.zeros(n, np.int64)
    truebins = np.zeros(n, np.int64)
    lookup = {}
    for sid in np.unique(states["stay_id"]):
        ii = np.flatnonzero(states["stay_id"] == sid)
        ii = ii[np.argsort(states["hour"][ii])]
        lookup[int(sid)] = (np.rint(states["hour"][ii] * 60).astype(int), states["x"][ii])
    for sid, gg in rows.groupby("stay_id", sort=False):
        times, xx = lookup[int(sid)]
        last_by_code = {}
        for idx, row in gg.iterrows():
            minute = int(round(row.hour * 60))
            p = np.searchsorted(times, minute, side="left") - 1
            if p >= 0:
                previous[idx] = xx[p]
                elapsed[idx] = max(1.0, (minute - times[p]) * 60.0)
            md = ce_meta[(row.event, "value")]
            truebins[idx] = 1 + np.searchsorted(md["fine_bin_edges"], row.value, side="right")
            if row.event in last_by_code:
                lv, lm = last_by_code[row.event]
                numeric[idx, 0] = 1.0
                numeric[idx, 3] = np.log1p((minute - lm) * 60.0) / 10.0
                numeric[idx, 6] = (lv - md["median"]) / md["iqr"]
                lastbins[idx] = 1 + np.searchsorted(md["fine_bin_edges"], lv, side="right")
            last_by_code[row.event] = (row.value, minute)
    state = torch.from_numpy(previous)
    delta_t = torch.from_numpy(elapsed)
    flow = state[:, 192:960].reshape(-1, 4, 192) * torch.exp(
        -delta_t[:, None, None] * W["core.flow.a"].exp()[None, None, :])
    base = state[:, :192] + norm(flow.mean(1), "core.out_norm")
    gapv = mlp(torch.log(delta_t.clamp_min(1.0))[:, None] / 10.0, "gap", "silu")
    context = base + gapv
    prior = mlp(context, "z_prior", "gelu")
    viewv = context + lin(prior[:, :32], "z_head")
    query = norm(viewv + W["type_emb.weight"][1], "query_norm")
    codes = torch.tensor([ce_code[e] for e in rows.event], dtype=torch.long)
    registerv = (lin(torch.from_numpy(numeric), "last_numeric") +
                 W["embed.bins.value.weight"][torch.from_numpy(lastbins)] +
                 W["embed.bins.rate.weight"][0] + W["last_status.weight"][0])
    withcode = query + W["embed.code.weight"][codes] + registerv
    phi = TF.relu(lin(withcode, "attr_heads.value_bin.0"))
    logitsv = lin(phi, "attr_heads.value_bin.2") + W["embed_transition.value_bin.weight"][
        codes * 17 + torch.from_numpy(lastbins)]
    probabilities = torch.softmax(logitsv, -1)
    centres = torch.tensor(np.stack([ce_centres[e] for e in rows.event]), dtype=torch.float32)
    return {"rows": rows, "phi": phi, "logits": logitsv, "probs": probabilities,
            "centres": centres, "truebin": truebins, "prior": prior,
            "withcode": withcode, "previous": previous}

ce_states = {k: np.concatenate([S[k], repair_S[k]], axis=0) for k in S}
ce_factual = pd.concat([
    pd.read_csv("/work/fwd_3/factual_original.csv"),
    pd.read_csv("/work/fwd_6/factual_original.csv")
], ignore_index=True).drop_duplicates(["stay_id", "hour", "event"])
ce_common_ids = np.intersect1d(ce_states["stay_id"], ce_factual.stay_id.unique())
ce_head = ce_make_heads(train[train.stay_id.isin(ce_common_ids)], ce_states)
ce_rows = ce_head["rows"].merge(
    ce_factual[["stay_id", "hour", "event", "predicted_mean", "nll"]],
    on=["stay_id", "hour", "event"], how="left", validate="one_to_one")
assert ce_rows.predicted_mean.notna().all()
ce_mu = (ce_head["probs"] * ce_head["centres"]).sum(1).numpy()
print("Calibration:", ce_rows.stay_id.nunique(), "stays,", len(ce_rows), "readings")
print("Approximate mean reconstruction RMSE:", np.sqrt(np.mean(
    (ce_mu - ce_rows.predicted_mean.to_numpy())**2)))

ce_si = pd.Index(stay_ids).get_indexer(ce_rows.stay_id)
ce_hour = ce_rows.hour.to_numpy()
ce_ins_family = ce_rows.event.str.startswith("glucose").to_numpy()
ce_ix = np.full(len(ce_rows), -1, int)
for r in np.flatnonzero(np.isin(ce_hour, list(repair_position))):
    ce_ix[r] = ce_si[r] * len(repair_position) + repair_position[int(ce_hour[r])]
ce_rd = np.full((len(ce_rows), 2), np.nan)
ce_valid = ce_ix >= 0
for k, key in enumerate((("glucose_finger", "insulin"),
                         ("map_arterial", "norepinephrine"))):
    lp = repair_robust[key]
    ii = np.flatnonzero(ce_valid)
    good = lp["mask"][ce_ix[ii]] & np.isfinite(lp["rd"][ce_ix[ii]])
    ce_rd[ii[good], k] = lp["rd"][ce_ix[ii[good]]]
ce_fitmask = ce_valid & (ce_hour >= 5) & np.where(
    ce_ins_family, np.isfinite(ce_rd[:, 0]), np.isfinite(ce_rd[:, 1]))
ce_target = np.zeros((len(ce_rows), 2))
ce_target[ce_ins_family, 0] = ce_rd[ce_ins_family, 0]
ce_target[~ce_ins_family, 1] = ce_rd[~ce_ins_family, 1]
ce_stay_split = (ce_si % 2).astype(int)
ce_decoder_results = []
ce_decoder_models = {}
for alpha in (1.0, 10.0, 100.0):
    trainmask = ce_fitmask & (ce_stay_split == 0)
    testmask = ce_fitmask & (ce_stay_split == 1)
    fit = Ridge(alpha=alpha).fit(ce_head["phi"].numpy()[trainmask], ce_target[trainmask])
    prediction = fit.predict(ce_head["phi"].numpy())
    ce_decoder_models[alpha] = fit
    for k, family in enumerate((ce_ins_family, ~ce_ins_family)):
        use = testmask & family
        ce_decoder_results.append((alpha, repair_drugs[k], int(use.sum()),
                                  r2_score(ce_target[use, k], prediction[use, k]),
                                  np.std(ce_target[use, k])))
print(pd.DataFrame(ce_decoder_results,
      columns=["alpha", "drug", "heldout_n", "heldout_r2", "innovation_sd"]).to_string(index=False))
with open("/work/notes.md", "a") as f:
    f.write("\n## Forward 6 / Run 29\nOriginal enriched training factual NLL 1.51284. "
            "Built value-head calibration features and stay-separated innovation-decoder tests; "
            "no edits yet. Combined only stays shared by state and factual exports.\n")
```