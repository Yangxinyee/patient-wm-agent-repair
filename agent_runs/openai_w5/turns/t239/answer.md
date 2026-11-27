Only 0.4% of recorded insulin infusions exceed the channel’s range when doubled, so saturation is unlikely to explain the bulk error. I’ll decompose the factual head inputs into transported exposure and centering. Strong cancellation would motivate a history-only centering test, though cancellation on factual records alone would not prove leakage.

```python
# Reconstruct head inputs from saved GPU states; no recurrent replay.
head_rows = train.loc[
    train.stay_id.isin(state_ids) &
    train.event.isin(sensor_neurons)
].sort_values(["stay_id", "hour", "event"]).copy()
_reg = head_rows.groupby(["stay_id", "event"], sort=False)
head_rows["last_value"] = _reg["value"].shift()
head_rows["last_hour"] = _reg["hour"].shift()
head_rows = head_rows.reset_index(drop=True)

_prev = np.full(len(head_rows), -1, dtype=np.int64)
for _sid, _ix in head_rows.groupby("stay_id", sort=False).groups.items():
    _si = np.flatnonzero(state_file["stay_id"] == _sid)
    _si = _si[np.argsort(state_file["hour"][_si])]
    _p = np.searchsorted(state_file["hour"][_si],
                         head_rows.loc[_ix, "hour"].to_numpy(), side="left") - 1
    _valid = _p >= 0
    _prev[np.asarray(_ix)[_valid]] = _si[_p[_valid]]
head_rows = head_rows.loc[_prev >= 0].reset_index(drop=True)
head_prev = _prev[_prev >= 0]

_dr = train.loc[train.stay_id.isin(state_ids) &
                train.event.isin(["insulin", "norepinephrine"])].copy()
_dr["read_hour"] = np.floor(_dr.hour).astype(int) + 1
_dp = _dr.pivot_table(index=["stay_id", "read_hour"], columns="event",
                     values="rate", aggfunc="sum").fillna(0)
head_rows = head_rows.merge(_dp.reset_index(), how="left",
                           left_on=["stay_id", "hour"],
                           right_on=["stay_id", "read_hour"], sort=False)
head_rows[["insulin", "norepinephrine"]] = head_rows[
    ["insulin", "norepinephrine"]].fillna(0)

def make_head_inputs212(weights, sf, rows, prev):
    x = torch.from_numpy(sf["x"][prev])
    h = x[:, 192:960].reshape(-1, 4, 192)
    dt = torch.tensor((rows.hour.to_numpy() - sf["hour"][prev]) * 3600,
                      dtype=torch.float32).clamp_min(1)
    hf = h * torch.exp(-dt[:, None, None] *
                       torch.exp(weights["core.flow.a"])[None, None, :])
    gap = linear205(F.gelu(linear205(torch.log1p(dt)[:, None] / 10,
                                     "gap.0", weights)), "gap.2", weights)
    ctx = x[:, :192] + norm205(hf.mean(1), "core.out_norm", weights) + gap
    zp = linear205(F.gelu(linear205(ctx, "z_prior.0", weights)),
                   "z_prior.2", weights)
    view = ctx + linear205(zp[:, :32], "z_head", weights)
    q = norm205(view + weights["type_emb.weight"][1], "query_norm", weights)

    ln = torch.zeros(len(rows), 8)
    vb = np.zeros(len(rows), np.int64)
    for event in sensor_neurons:
        ix = np.flatnonzero(rows.event.to_numpy() == event)
        spec = num_spec[(event, "value")]
        lv = rows.last_value.to_numpy()[ix]
        present = np.isfinite(lv)
        ln[ix, 0] = torch.tensor(present, dtype=torch.float32)
        ln[ix, 3] = torch.tensor(np.where(present,
            np.clip((lv - spec["median"]) / spec["iqr"], -10, 10), 0),
            dtype=torch.float32)
        age = (rows.hour.to_numpy()[ix] - rows.last_hour.to_numpy()[ix]) * 3600
        ln[ix, 6] = torch.tensor(np.where(present, np.log1p(age) / 10, 0),
                                dtype=torch.float32)
        vb[ix] = np.where(present,
            np.searchsorted(spec["fine_bin_edges"], lv, side="left") + 1, 0)
    codes = torch.tensor([code_id[e] for e in rows.event], dtype=torch.long)
    wc = (q + weights["embed.code.weight"][codes]
          + linear205(ln, "last_numeric", weights)
          + weights["embed.bins.value.weight"][torch.from_numpy(vb)]
          + weights["embed.bins.rate.weight"][0])
    return wc, q, ln, vb, zp

head_wc, head_q, head_ln, head_vb, head_zp = make_head_inputs212(
    W, state_file, head_rows, head_prev)

print("=== EXPOSURE / CENTERING DECOMPOSITION ===")
head_audit = {}
for _event, (_j, _control) in sensor_neurons.items():
    _ix = np.flatnonzero(head_rows.event.to_numpy() == _event)
    _dw = W["attr_heads.value_bin.0.weight"][_j] - W["attr_heads.value_bin.0.weight"][_control]
    _db = W["attr_heads.value_bin.0.bias"][_j] - W["attr_heads.value_bin.0.bias"][_control]
    _ex = (head_wc[_ix][:, priv] @ _dw[priv]).numpy()
    _ce = -(head_wc[_ix][:, ordinary_idx] @ _dw[ordinary_idx] + _db).numpy()
    _drug = "insulin" if _event.startswith("glucose") else "norepinephrine"
    _rate = head_rows.loc[_ix, _drug].to_numpy()
    _slope = np.linalg.lstsq(np.column_stack([np.ones(len(_ix)), _rate]),
                            _ex, rcond=None)[0]
    _g = linear205(head_wc[_ix], "attr_heads.value_bin.0", W)[:, _control]
    head_audit[_event] = {"ix": _ix, "exposure": _ex, "center": _ce,
                          "rate": _rate, "slope": _slope}
    print(_event, "n", len(_ix),
          "exposure intercept/slope", np.round(_slope, 5),
          "exposure-center correlation", round(float(np.corrcoef(_ex, _ce)[0, 1]), 4),
          "SD exposure/center/residual", np.round(
              [np.std(_ex), np.std(_ce), np.std(_ex-_ce)], 5),
          "control gate 10/50/90%", np.round(
              np.quantile(_g.numpy(), [.1, .5, .9]), 3))

print("\n=== AVAILABLE FACTUAL MATCHES ===")
for _folder in sorted(Path("/work").glob("fwd_*")):
    _p = _folder / "factual_original.csv"
    if _p.exists():
        _df = pd.read_csv(_p)
        if "stay_id" in _df and len(np.intersect1d(
                _df.stay_id.unique(), state_ids)):
            print(_folder.name, _df.shape, _df.columns.tolist(),
                  "overlap", len(np.intersect1d(_df.stay_id.unique(), state_ids)))
```