The fresh rollout confirms all four edited-model shifts are exactly zero on another 100 stays. Together with the 400-stay factual check, this supports submitting the current three-tensor edit.

The records support a near-zero target in the evaluated window, not proof of exactly zero true effects. I’ll record the final checks and make no further changes. A nonzero saved rollout shift or an NLL increase above 0.005 would stop submission.

```python
final_rollout_tables = []
for k in [4, 8]:
    df = pd.read_csv(f"/work/fwd_{k}/dose_edit_edited.csv")
    assert np.isfinite(df["shift"]).all()
    assert float(df["shift"].abs().max()) == 0.0
    reached = df["reached"]
    if reached.dtype != bool:
        reached = reached.astype(str).str.lower().eq("true")
    final_rollout_tables.append(df.loc[reached].copy())

final_rollout_table = pd.concat(final_rollout_tables, ignore_index=True)
assert not final_rollout_table.duplicated(
    ["stay_id", "hour", "edit"]
).any()

final_readouts = {}
for label, group in final_rollout_table.groupby("edit"):
    final_readouts[label] = {
        "mean_shift": float(group["shift"].mean()),
        "maximum_absolute_shift": float(group["shift"].abs().max()),
        "eligible_stays": int(group["stay_id"].nunique()),
        "eligible_stay_hours": len(group)
    }
assert len(final_readouts) == 4

assert audit_summary["stays"] == 400
assert audit_summary["delta_nll"] < 0.005
assert set(edited()) == {
    "embed.numeric.weight",
    "embed.bins.dose.weight",
    "embed.bins.rate.weight"
}

w = load("embed.numeric.weight")
w0 = load_original("embed.numeric.weight")
assert torch.equal(w[:, :2], w0[:, :2])
assert torch.count_nonzero(w[:, [2, 4]]).item() == 0
for name in ["embed.bins.dose.weight", "embed.bins.rate.weight"]:
    w = load(name)
    w0 = load_original(name)
    assert torch.equal(w[0], w0[0])
    assert torch.equal(w[1:], w[1:2].expand_as(w[1:]))

final_report = {
    "edited_tensors": sorted(edited()),
    "readouts": final_readouts,
    "factual_validation": audit_summary,
    "python_runs": 29,
    "forward_passes": 8,
    "interpretation": (
        "Dose-magnitude invariance is the conservative repair supported "
        "by adjusted training-record analyses in the evaluated window. "
        "Small nonzero true effects remain possible; significant all-hours "
        "NE associations do not establish a universal null effect."
    )
}
print(json.dumps(final_report, indent=2))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Python run 29 / forward passes 8: final submission checks\n"
        "- Fresh exact rollout on tuning stays 2020000300–2020000399: "
        "all four edited-model mean shifts exactly zero; edit hash e723af07c9.\n"
        "- Independently checked saved rollout CSVs from forwards 4 and 8: "
        "every shift is exactly zero, with no duplicated stay-hour/edit rows.\n"
        "- Factual validation over all 400 tuning stays / 19,413 readings: "
        "original 1.48944949, edited 1.48975656; increase +0.00030707 nats, "
        "below the +0.005 limit.\n"
        "- Submitting only the three existing embedding edits: zero numeric "
        "dose/rate magnitude columns with training-mean centering in presence "
        "columns; frequency-weighted constant positive dose/rate-bin rows. "
        "Measurement columns, absent-bin rows, identity and timing are retained.\n"
        "- Conservative target: near-zero dose-magnitude effects in the "
        "evaluated window. Do not claim observational records prove exact "
        "zero effects or absence of early-hour effects.\n"
        "- Final report: " + json.dumps(final_report) + "\n"
        "- No further weight changes planned.\n"
    )
```
