The exact factual increase is **0.006724**. The archive confirms that this trial strengthened the insulin correction substantially. I’ll restore the previously validated fallback and compare past measured losses and rollouts before choosing another trial. Any further increase must improve dose responses without exhausting the factual-fit margin.

```python
_r280_original = np.asarray(load_original("core.b_in.weight"), dtype=np.float32)
_r280_current = np.asarray(load("core.b_in.weight"), dtype=np.float32)
with np.load("/work/joint278.npz") as _r280_z:
    _r280_delta278 = _r280_z["delta"].copy()
_r280_error = float(np.max(np.abs(
    _r280_current - (_r280_original + _r280_delta278))))
print("Current/archive agreement:", _r280_error)
assert _r280_error < 1e-4, "Unexpected archive convention; no restoration performed."

with np.load("/work/joint274.npz") as _r280_z:
    _r280_safe = {k: _r280_z[k].copy() for k in _r280_z.files}
save_edit("core.b_in.weight", _r280_original + _r280_safe["delta"])
print("Restored joint274; decomposition residual norm:",
      float(np.linalg.norm(_r280_safe["delta"]
                           - _r280_safe["insulin_delta"]
                           - _r280_safe["ne_delta"])))

_r280_rows = []
_r280_dose_files = []
for _r280_n in range(1, 41):
    _r280_dir = "/work/fwd_" + str(_r280_n)
    _r280_op = _r280_dir + "/factual_original.csv"
    _r280_ep = _r280_dir + "/factual_edited.csv"
    if os.path.exists(_r280_op) and os.path.exists(_r280_ep):
        _r280_o = pd.read_csv(_r280_op)
        _r280_e = pd.read_csv(_r280_ep)
        _r280_rows.append({
            "forward": _r280_n,
            "stays": _r280_o.stay_id.nunique(),
            "first_stay": _r280_o.stay_id.min(),
            "last_stay": _r280_o.stay_id.max(),
            "n": len(_r280_o),
            "nll_delta": float(_r280_e.nll.mean() - _r280_o.nll.mean())
        })
    _r280_dp = _r280_dir + "/dose_edit_edited.csv"
    if os.path.exists(_r280_dp):
        _r280_dose_files.append((_r280_n, _r280_dp))

print("\nPAIRED FACTUAL HISTORY")
print(pd.DataFrame(_r280_rows).to_string(index=False))
print("\nROLLOUT FILES:", [n for n, p in _r280_dose_files])
_r280_dose = pd.read_csv("/work/fwd_35/dose_edit_edited.csv")
print("Rollout schema:", list(_r280_dose.columns))
print(_r280_dose.head(3).to_string(index=False))

_r280_notes = open("/work/notes.md").read()
_r280_lines = [s for s in _r280_notes.splitlines()
               if "forward" in s.lower()
               and any(t in s.lower() for t in
                       ["insulin", "ne2", "nehalf", "joint", "factual cost", "factual delta"])]
print("\nRECENT MEASURED FRONTIER NOTES")
print("\n".join(_r280_lines[-24:]))

with open("/work/notes.md", "a") as _r280_f:
    _r280_f.write(
        "\n## Run 280: restored validated fallback\n"
        "Exact forward40 joint278 factual delta=.00672373499 on tuning100..199. "
        "Verified archive convention and restored joint274 "
        "(I=.684, blend=.60, NE=.36), whose all400 factual delta=.002951029. "
        "Reviewing measured factual/rollout frontier before further trials.\n"
        "Python runs: 280; forwards: 40.\n"
    )
```