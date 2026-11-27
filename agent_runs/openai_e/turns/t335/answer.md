Past edits substantially improved insulin responses, but norepinephrine responses remain too small. Most of the stronger trial’s added factual loss came from glucose predictions. I’ll test a reduced version of the earlier insulin direction and resolve the averaging discrepancy before comparing effects. It must beat the fallback’s response accuracy while respecting the loss limit.

```python
print("FORWARD35 AGGREGATION CHECK")
_r282_d = pd.read_csv("/work/fwd_35/dose_edit_edited.csv")
_r282_reached = _r282_d[
    _r282_d.reached.astype(str).str.lower().isin(["true", "1"])
    & _r282_d.hour.between(13, 24)
]
_r282_agg = pd.DataFrame({
    "all_rows": _r282_d.groupby("edit").shift.mean(),
    "reached_hours": _r282_reached.groupby("edit")["shift"].mean(),
    "equal_stays": _r282_reached.groupby(["edit", "stay_id"])["shift"]
        .mean().groupby("edit").mean()
})
print(_r282_agg.to_string())
print("Forward35 files:", os.listdir("/work/fwd_35"))
print("Hours:", sorted(_r282_d.hour.unique().tolist()))

with np.load("/work/joint236.npz") as _r282_z:
    _r282_source = {k: _r282_z[k].copy() for k in _r282_z.files}
_r282_i_scale, _r282_ne_scale = 0.78, 0.20
_r282_i = (_r282_i_scale / float(_r282_source["insulin_scale"])) * _r282_source["insulin_delta"]
_r282_ne = (_r282_ne_scale / float(_r282_source["ne_scale"])) * _r282_source["ne_delta"]
_r282_residual = (_r282_source["delta"] - _r282_source["insulin_delta"]
                  - _r282_source["ne_delta"])
_r282_delta = (_r282_residual + _r282_i + _r282_ne).astype(np.float32)
save_edit("core.b_in.weight", _r280_original + _r282_delta)
np.savez("/work/joint282.npz",
         delta=_r282_delta, insulin_delta=_r282_i, ne_delta=_r282_ne,
         insulin_scale=_r282_i_scale, insulin_blend=0.35, ne_scale=_r282_ne_scale)
print("Installed unaccepted candidate A: insulin=.78, blend=.35, NE=.20")
print("Delta norm:", float(np.linalg.norm(_r282_delta)), "edits:", edited())

with open("/work/notes.md", "a") as _r282_f:
    _r282_f.write(
        "\n## Run 282: reduced earlier insulin-direction trial\n"
        "Past dose summaries require aggregation audit: equal-stay means differ "
        "from reported forward summaries. Printed all-row, reached-hour and "
        "equal-stay means to identify the evaluator's convention. "
        "Installed unaccepted candidate A: I=.78, blend=.35, NE=.20, "
        "using archived joint236 corrections and original b_in; no gradients. "
        "Saved joint282.npz. Joint274 remains validated fallback. "
        "Next paired factual and dose validation on tuning100..199.\n"
        "Python runs: 282; forwards: 40.\n"
    )
```