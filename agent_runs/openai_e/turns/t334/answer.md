The fallback was restored exactly, and its factual loss increase was below 0.005 in all four tuning cohorts. Next I’ll compare past dose responses and event-specific losses to choose a better allocation of that margin. A useful trial should improve insulin’s measured response without sacrificing too much norepinephrine accuracy.

```python
_r281_rollout_rows = []
for _r281_n, _r281_path in _r280_dose_files:
    for _r281_model in ["original", "edited"]:
        _r281_p = "/work/fwd_%d/dose_edit_%s.csv" % (_r281_n, _r281_model)
        if not os.path.exists(_r281_p):
            continue
        _r281_d = pd.read_csv(_r281_p)
        _r281_use = _r281_d[
            _r281_d["reached"].astype(str).str.lower().isin(["true", "1"])
            & _r281_d["hour"].between(13, 24)
        ]
        _r281_means = (
            _r281_use.groupby(["edit", "stay_id"])["shift"].mean()
            .groupby("edit").mean()
        )
        _r281_rollout_rows.append({
            "forward": _r281_n, "model": _r281_model,
            "stays": _r281_d.stay_id.nunique(),
            **_r281_means.to_dict()
        })
print("REACHED-HOUR, EQUAL-STAY ROLLOUT MEANS")
print(pd.DataFrame(_r281_rollout_rows).round(5).to_string(index=False))

_r281_event_rows = []
for _r281_n, _r281_label in [(33, "joint236"), (34, "joint268"),
                             (36, "joint274"), (40, "joint278")]:
    _r281_o = pd.read_csv("/work/fwd_%d/factual_original.csv" % _r281_n)
    _r281_e = pd.read_csv("/work/fwd_%d/factual_edited.csv" % _r281_n)
    assert _r281_o[["stay_id", "hour", "event", "observed"]].equals(
        _r281_e[["stay_id", "hour", "event", "observed"]])
    _r281_diff = _r281_o[["event"]].copy()
    _r281_diff["delta"] = _r281_e.nll.to_numpy() - _r281_o.nll.to_numpy()
    for _r281_event, _r281_group in _r281_diff.groupby("event"):
        _r281_event_rows.append({
            "candidate": _r281_label, "event": _r281_event,
            "mean_delta": _r281_group.delta.mean(),
            "overall_contribution": _r281_group.delta.sum() / len(_r281_diff)
        })
print("\nFACTUAL COST BY EVENT, SAME COHORT")
print(pd.DataFrame(_r281_event_rows).round(7).to_string(index=False))

print("\nEARLY RECORD-IDENTIFICATION NOTEBOOK")
print(_r280_notes[:6500])

with open("/work/notes.md", "a") as _r281_f:
    _r281_f.write(
        "\n## Run 281: measured response/cost frontier\n"
        "Joint274 remains installed. Recomputed past rollout summaries using "
        "only reached hours13..24, averaging within stay before across stays. "
        "Compared paired event-specific factual costs on the identical "
        "tuning100..199 cohort and recovered early record-identification evidence. "
        "No weights changed.\nPython runs: 281; forwards: 40.\n"
    )
```