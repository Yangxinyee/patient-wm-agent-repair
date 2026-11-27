Latent sampling explains only part of the discrepancy, and a glucose bias remains. I’ll broaden the last-register layout search and separate first readings from later ones. A match on later readings is particularly important because the dose evaluation begins after twelve hours of history.

```python
import itertools, time
t0 = time.time()

# Locate systematic errors before trying a correction.
head_all_check = local_value_head(
    head_data, head_cfg, bin_side=head_bin_side,
    zmode="prior_mc", nsamples=32, return_features=True
)
audit = head_data[["event", "hour", "has_last"]].copy()
audit["mean_error"] = head_all_check["mean"] - head_data.predicted_mean.to_numpy()
audit["nll_error"] = head_all_check["nll"] - head_data.nll.to_numpy()
audit["phase"] = np.where(audit.has_last.eq(0), "no_previous_same_code",
                         np.where(audit.hour.le(0), "early_with_last", "later_with_last"))
print("=== RECONSTRUCTION ERROR BY HISTORY ===")
print(audit.groupby(["event", "phase"]).agg(
    n=("mean_error", "size"),
    mean_bias=("mean_error", "mean"),
    mean_rmse=("mean_error", lambda v: np.sqrt(np.mean(v**2))),
    nll_bias=("nll_error", "mean"),
    nll_rmse=("nll_error", lambda v: np.sqrt(np.mean(v**2)))
).round(4).to_string())

# Search all one-flag/two-flag register layouts using an MC-averaged query.
hh = head_data.iloc[test_rows]
mc = local_value_head(hh, head_cfg, bin_side="left", zmode="prior_mc",
                      nsamples=128, return_features=True)
n0 = len(hh)
hl1 = torch.tensor(hh.has_last.to_numpy(), dtype=torch.float32)
zl1 = torch.tensor(hh.z_last.to_numpy(), dtype=torch.float32)
age_sec = torch.log1p(torch.tensor(hh.last_age_s.to_numpy(), dtype=torch.float32)) / 10
age_hour = torch.log1p(torch.tensor(hh.last_age_s.to_numpy(), dtype=torch.float32) / 3600) / 10
cur_last = (
    zl1[:, None] * lnproj[int(head_cfg["zcol"])]
    + age_sec[:, None] * lnproj[int(head_cfg["acol"])]
    + hl1[:, None] * lnproj[list(head_cfg["flags"])].sum(0)
)
fixed_pre = torch.tensor(mc["pre"]) - cur_last
cid1 = torch.tensor(hh.code_id.to_numpy(), dtype=torch.long)
lb1 = torch.tensor(hh.last_bin.to_numpy(), dtype=torch.long)
ob1 = torch.tensor(hh.observed_bin.to_numpy(), dtype=torch.long)
tr1 = hw("embed_transition.value_bin.weight")[cid1 * 17 + lb1]
cc1 = torch.tensor(np.array([
    [0.0] + VOCAB["value_bin_centres"][e] for e in hh.event
]), dtype=torch.float32)

register_rows = []
for zc in range(8):
    for ac in range(8):
        if ac == zc:
            continue
        other = [j for j in range(8) if j not in (zc, ac)]
        for flag_count in (1, 2):
            for flags in itertools.combinations(other, flag_count):
                for age_mode, ax in (("seconds_div10", age_sec), ("hours_div10", age_hour)):
                    pp0 = fixed_pre + zl1[:, None] * lnproj[zc] + ax[:, None] * lnproj[ac]
                    pp0 += hl1[:, None] * lnproj[list(flags)].sum(0)
                    for va in ("gelu", "silu", "relu"):
                        logits0 = TF.linear(
                            act_fn(pp0, va), hw("attr_heads.value_bin.2.weight"),
                            hw("attr_heads.value_bin.2.bias")
                        ) + tr1
                        probs0 = torch.softmax(logits0, -1)
                        means0 = (probs0 * cc1).sum(-1).numpy()
                        nll0 = -torch.log(probs0[torch.arange(n0), ob1].clamp_min(1e-12)).numpy()
                        mr = float(np.sqrt(np.mean((means0 - hh.predicted_mean.to_numpy())**2)))
                        nr = float(np.sqrt(np.mean((nll0 - hh.nll.to_numpy())**2)))
                        register_rows.append(dict(
                            zcol=zc, acol=ac, flags=flags, age_mode=age_mode,
                            value_activation=va, mean_rmse=mr, nll_rmse=nr, score=mr+3*nr
                        ))
register_search = pd.DataFrame(register_rows).sort_values("score")
print("\n=== FULL REGISTER SEARCH ===")
print(register_search.head(12).to_string(index=False))
print("seconds:", round(time.time()-t0, 1))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 18: expanded register audit\n"
            + audit.groupby(["event", "phase"]).mean(numeric_only=True).to_json() + "\n"
            + register_search.head(5).to_json(orient="records") + "\n"
            "Head reconstruction remains provisional until systematic errors are explained.\n")
```