The record-derived kernels suggest delayed glucose lowering from insulin and a shorter MAP increase from norepinephrine. Their provisional rollout targets remain much larger than the original model’s. I’ll now compare factual prediction errors with dose innovations; consistent drug-related residuals would support a targeted head correction, while near-zero residuals would argue against it.

```python
import io
import dis

print("=== HEAD CACHE QUALITY ===")
_r37_phi = phij.detach().cpu().numpy() if torch.is_tensor(phij) else np.asarray(phij)
_r37_p = pj.detach().cpu().numpy() if torch.is_tensor(pj) else np.asarray(pj)
print("features:", _r37_phi.shape, "probability row sums:",
      np.min(_r37_p.sum(1)), np.max(_r37_p.sum(1)))
print("wide index:", wide.index.names, list(wide.index[:3]))
print("history columns:", list(Xhist.columns))
print("vocabulary value centres:", str(VOCAB["value_bin_centres"])[:1800])
print("numeric specifications:", numeric_specs)

repair_aligned = {}
repair_slope_diagnostics = {}
for _drug, _event in (
    ("insulin", "glucose_finger"),
    ("norepinephrine", "map_arterial")
):
    _dc = dml_cache[_drug]
    _source_index = wide.index[np.asarray(_dc["inds"], dtype=int)]
    _lookup = pd.DataFrame({
        "dose": _dc["d"], "dose_expected": _dc["pred_d"],
        "rd": _dc["rd"], "outcome_expected": _dc["pred_y"],
        "ry": _dc["ry"]
    }, index=_source_index)
    _target_index = pd.MultiIndex.from_arrays([
        head_meta["stay_id"].to_numpy(),
        head_meta["hour"].to_numpy() - 1
    ], names=wide.index.names)
    _matched = _lookup.reindex(_target_index).reset_index(drop=True)
    _matched["event"] = head_meta["event"].to_numpy()
    _matched["observed"] = head_meta["observed"].to_numpy()
    _matched["tool_mean"] = head_meta["tool_mean"].to_numpy()
    _matched["stay_id"] = head_meta["stay_id"].to_numpy()
    _matched["hour"] = head_meta["hour"].to_numpy()
    repair_aligned[_drug] = _matched
    _ok = (_matched["event"].eq(_event) &
           _matched["rd"].notna()).to_numpy()
    _z = _matched.loc[_ok]
    _den = np.sum(_z["rd"].to_numpy() ** 2)
    _slopes = {
        "n": len(_z),
        "innovation_SD": float(_z["rd"].std()),
        "record_slope": float(np.sum(
            _z["rd"] * (_z["observed"] - _z["outcome_expected"])
        ) / _den),
        "model_slope": float(np.sum(
            _z["rd"] * (_z["tool_mean"] - _z["outcome_expected"])
        ) / _den),
        "missing_slope": float(np.sum(
            _z["rd"] * (_z["observed"] - _z["tool_mean"])
        ) / _den)
    }
    repair_slope_diagnostics[_drug] = _slopes
    print(_drug, _slopes)

print("=== FEATURE FUNCTION DEFINITIONS ===")
for _name in ("rh_register", "rh_eval"):
    _fn = globals().get(_name)
    if callable(_fn):
        _buf = io.StringIO()
        dis.dis(_fn, file=_buf)
        print(_name, "defaults:", _fn.__defaults__)
        print(_buf.getvalue()[:2100])

print("=== RECONSTRUCTION SETTINGS ===")
print("rh_chosen:", rh_chosen)
print("candidate config:", candidate.get("config"))
print("State dimensions:", {k: np.shape(v) for k, v in states0.items()})
```