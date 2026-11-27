The script hit a pandas column-name error before changing any weights, so the validated fallback is still installed. I’ll fix the diagnostic and install the planned reduced trial. Lower strength should reduce factual disturbance; paired factual and rollout results will determine whether it is worth keeping.

```python
print("FORWARD35 AGGREGATION CHECK")
_r283_d = pd.read_csv("/work/fwd_35/dose_edit_edited.csv")
_r283_reached = _r283_d[
    _r283_d["reached"].astype(str).str.lower().isin(["true", "1"])
    & _r283_d["hour"].between(13, 24)
]
_r283_agg = pd.DataFrame({
    "all_rows": _r283_d.groupby("edit")["shift"].mean(),
    "reached_hours": _r283_reached.groupby("edit")["shift"].mean(),
    "equal_stays": _r283_reached.groupby(["edit", "stay_id"])["shift"]
        .mean().groupby("edit").mean()
})
print(_r283_agg.to_string())
print("Forward35 files:", os.listdir("/work/fwd_35"))
print("Hours:", sorted(_r283_d["hour"].unique().tolist()))

with np.load("/work/joint236.npz") as _r283_z:
    _r283_source = {k: _r283_z[k].copy() for k in _r283_z.files}
_r283_i_scale, _r283_ne_scale = 0.78, 0.20
_r283_i = (_r283_i_scale / float(_r283_source["insulin_scale"])) * _r283_source["insulin_delta"]
_r283_ne = (_r283_ne_scale / float(_r283_source["ne_scale"])) * _r283_source["ne_delta"]
_r283_residual = (_r283_source["delta"] - _r283_source["insulin_delta"]
                  - _r283_source["ne_delta"])
_r283_delta = (_r283_residual + _r283_i + _r283_ne).astype(np.float32)
save_edit("core.b_in.weight", _r280_original + _r283_delta)
np.savez("/work/joint283.npz",
         delta=_r283_delta, insulin_delta=_r283_i, ne_delta=_r283_ne,
         insulin_scale=_r283_i_scale, insulin_blend=0.35, ne_scale=_r283_ne_scale)
print("Installed unaccepted A: insulin=.78, blend=.35, NE=.20")
print("Delta norm:", float(np.linalg.norm(_r283_delta)), "edits:", edited())

with open("/work/notes.md", "a") as _r283_f:
    _r283_f.write(
        "\n## Runs 282-283: corrected diagnostic and candidate A\n"
        "Run282 failed on pandas GroupBy.shift name collision before editing. "
        "Run283 uses explicit column indexing, audits rollout aggregation, "
        "and installs unaccepted A: I=.78, blend=.35, NE=.20. "
        "Closed-form recombination of joint236 corrections; no gradients. "
        "Saved joint283.npz. Joint274 remains the validated fallback. "
        "Next paired factual and dose validation on tuning100..199.\n"
        "Python runs: 283; forwards: 40.\n"
    )
```