The branch is active: posterior changes are substantial, while output changes are about four orders of magnitude smaller. That points to weak transmission rather than inactive gates. I’ll install a reversible 200/200-gain probe and calculate six-lag response projections from the records. If the stronger probe still barely affects rollouts—or violates factual fit—I’ll abandon simple amplification.

```python
r180_candidate4 = {
    n: load(n).clone() for n in
    ("z_post.0.weight", "z_post.0.bias", "z_post.2.weight", "z_post.2.bias")
}
rp_current_weights = rp_install(200.0, 200.0)
rp_current_gains = (200.0, 200.0)

r180_rollout = pd.read_csv("/work/fwd_13/dose_edit_original.csv")
r180_projection = {}
print("=== SIX-LAG RECORD-BASED RESPONSE PROJECTIONS (NOT ORACLE EFFECTS) ===")
for _drug, _edit in [("insulin", "insulin_x2"), ("norepinephrine", "ne_x2")]:
    _inf = r179_tuning[
        (r179_tuning.event == _drug) & r179_tuning.rate.notna() &
        (r179_tuning.hour >= 12) & (r179_tuning.hour < 24)
    ].copy()
    _inf["start_h"] = np.floor(_inf.hour).astype(int)
    _rates = _inf.groupby(["stay_id", "start_h"]).rate.sum().to_dict()
    _rows = r180_rollout[(r180_rollout.edit == _edit) & r180_rollout.reached].copy()
    _k = kernel_recovery[_drug]
    _rows["projection"] = [
        sum(_k[j] * _rates.get((int(sid), int(np.floor(h)) - 1 - j), 0.0)
            for j in range(len(_k)) if int(np.floor(h)) - 1 - j >= 12)
        for sid, h in zip(_rows.stay_id, _rows.hour)
    ]
    r180_projection[_drug] = _rows
    _mean = float(_rows.projection.mean())
    print(_drug, "reached hours", len(_rows),
          "x2 projection", round(_mean, 4),
          "half projection", round(-0.5 * _mean, 4),
          "original x2", round(float(_rows["shift"].mean()), 4))
    print("hourly projection:",
          _rows.groupby("hour").projection.mean().round(3).to_dict())

print("\n=== STRONG-PROBE OFFLINE SIZE ===")
_d180 = _post178(rp_X) - _post178(rp_X, True)
print("mean-coordinate RMS", float(np.sqrt(np.mean(_d180[:,:32]**2))))
print("mean-coordinate max abs", float(np.max(np.abs(_d180[:,:32]))))
print("gains", rp_current_gains, "edits", edited())

print("\n=== KERNEL METHODOLOGY NOTES ===")
_txt180 = Path("/work/notes.md").read_text()
_pos180 = [m.start() for m in __import__("re").finditer(
    r"(?i)(distributed.lag|impulse.response|horizon|six.lag|lag.kernel)", _txt180)]
for _p in _pos180[:7]:
    print(_txt180[max(0,_p-160):_p+450].replace("\n", " "))

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 180\n"
             "Actual gain-4 branch changes posterior RMS .07939 but out RMS only 6.75e-6 "
             "(drug-bundle out about 6–7e-7): active posterior, weak transmission. Installed "
             "reversible gain-200/200 probe with original numeric embedding. "
             "Six-lag observational x2 projections on enriched tuning reached hours: " +
             repr({d: float(r.projection.mean()) for d,r in r180_projection.items()}) +
             ". These are inferred projections, not supplied truth. "
             "Counts: 180 Python runs, 14 forwards.\n")
```
