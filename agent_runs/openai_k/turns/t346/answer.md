The edited rollout predicts insulin x2/half shifts of −10.879/+4.850, close to the retained record-based targets. Norepinephrine improved to +0.854/−0.257 but remains substantially under-responsive. I’ll calculate exact 400-stay factual fit and uncertainty before checking another rollout batch; an upper uncertainty bound below 0.005 would support retaining this candidate.

```python
margin_parts, margin_nll_rows = [], []
for folder in (37, 38, 40, 41):
    a = pd.read_csv(f"/work/fwd_{folder}/factual_original.csv")
    b = pd.read_csv(f"/work/fwd_{folder}/factual_edited.csv")
    keys = ["stay_id", "hour", "event", "observed"]
    assert a[keys].equals(b[keys])
    q = a[keys].copy()
    q["original_nll"] = a.nll.to_numpy()
    q["edited_nll"] = b.nll.to_numpy()
    q["delta_nll"] = q.edited_nll - q.original_nll
    margin_parts.append(q)
    margin_nll_rows.append((folder, len(q), float(q.delta_nll.mean())))
margin_paired400 = pd.concat(margin_parts, ignore_index=True)
assert margin_paired400.stay_id.nunique() == 400
assert not margin_paired400.duplicated(["stay_id", "hour", "event"]).any()
margin_stay_stats = margin_paired400.groupby("stay_id").delta_nll.agg(["sum", "count"])
margin_rng = np.random.default_rng(291)
margin_draws = margin_rng.integers(0, len(margin_stay_stats), size=(20000, len(margin_stay_stats)))
margin_boot = (
    margin_stay_stats["sum"].to_numpy()[margin_draws].sum(axis=1) /
    margin_stay_stats["count"].to_numpy()[margin_draws].sum(axis=1)
)
margin_quantiles = np.quantile(margin_boot, [0.025, 0.5, 0.95, 0.975])
print("Per-batch exact results:", margin_nll_rows)
print("400-stay NLL:", margin_paired400[["original_nll", "edited_nll", "delta_nll"]].mean().to_dict())
print("Readings:", len(margin_paired400), "bootstrap [2.5%,50%,95%,97.5%]:", margin_quantiles)
print("By event:", margin_paired400.groupby("event").delta_nll.agg(["count", "mean"]).to_dict())

margin_archive = np.load("/work/margin_candidate.npz")
margin_current = np.asarray(load("core.b_in.weight"))
assert np.array_equal(margin_current, margin_archive["core_b_in"])
margin_sv = np.linalg.svd(margin_current - margin_archive["original"], compute_uv=False)
print("Largest correction singular values:", margin_sv[:8])

for name in ("r244_targets", "targets158", "effects63", "target_lines"):
    if name in globals():
        print("RECORD TARGET", name, str(globals()[name])[:1200])

margin_paired400.to_csv("/work/margin_paired400.csv", index=False)
np.savez("/work/margin_validated400.npz",
         core_b_in=margin_current,
         pooled_value_nll_delta=margin_paired400.delta_nll.mean(),
         bootstrap_quantiles=margin_quantiles)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 291: margin candidate complete factual validation and first rollout\n"
            f"- Factual files fwd_37/38/40/41: exact pooled 400-stay delta "
            f"{margin_paired400.delta_nll.mean():.9f} over {len(margin_paired400)} readings.\n"
            f"- Stay-bootstrap quantiles [2.5%,50%,95%,97.5%]: {margin_quantiles.tolist()}.\n"
            "- Forward count 41, file fwd_42: insulin x2/half -10.879/+4.850; NE x2/half +0.854/-0.257.\n"
            "- Insulin is near retained record-convolution targets; NE remains materially under-responsive.\n"
            "- Archived exact validated weights; no weights changed, gradients, or weight optimizer.\n"
            "- Python runs: 291; forward passes: 41 (one failed request created a folder-number offset).\n")
```