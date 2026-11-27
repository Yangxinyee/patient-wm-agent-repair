The original prediction errors have dose-residual slopes in the expected missing-effect directions, but this small sample gives wide uncertainty. More importantly, my new full-sample head reconstruction is worse than the cached one, and its innovation decoders fail held-out tests. I’ll resolve that discrepancy before making an edit; otherwise these decoder results are not trustworthy.

```python
import dis
import io
import contextlib

print("=== EXACT REGISTER AND ACTIVATION FUNCTIONS ===")
for nm in ["rh_register", "rc_act"]:
    fun = globals()[nm]
    try:
        print(inspect.getsource(fun))
    except Exception:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dis.dis(fun)
        print(nm, buf.getvalue()[:4200])

print("\n=== CACHED RECONSTRUCTION METADATA ===")
print(rh_meta.head(3).to_string(index=False))
print("rh_nf first rows:", rh_nf[:3])
print("rh_last vs metadata last_z:",
      float((rh_last - torch.tensor(
          rh_meta.last_z.to_numpy(), dtype=torch.float32)).abs().max()))

key_cols = ["stay_id", "hour", "event"]
lookup = wr_meta.reset_index().set_index(key_cols)["index"]
wr_overlap = lookup.reindex(
    pd.MultiIndex.from_frame(rh_meta[key_cols])).to_numpy(dtype=int)
print("Register max discrepancy:", float(
    (wr_nf[wr_overlap] - rh_nf).abs().max()))
print("Fixed component max discrepancy:", float(
    (wr_fixed[wr_overlap] - rh_fixed
     - rc_lin(rh_nf, "last_numeric")).abs().max()))
print("Base versus rh_baseview:", float(
    (wr_base[wr_overlap] - rh_baseview).square().mean().sqrt()))
print("Code IDs:", code_ids)
print("Mean error by event:")
for ev, ix in wr_meta.groupby("event").groups.items():
    ix = np.array(list(ix))
    err_ev = wr_mean[ix] - wr_meta.tool_mean.to_numpy()[ix]
    print(ev, "RMSE", round(float(np.sqrt(np.mean(err_ev**2))), 3),
          "bias", round(float(err_ev.mean()), 3),
          "largest", round(float(np.max(np.abs(err_ev))), 3))
print("Mean error on cached 1000:", float(np.sqrt(np.mean(
    (wr_mean[wr_overlap] - rh_meta.tool_mean.to_numpy())**2))))

print("\n=== PRIOR CANDIDATE STRUCTURE ===")
for trial in sorted(rh_prior_trials, key=lambda x: x[0])[:3]:
    print([(tuple(x.shape), float(x.float().mean()))
           if isinstance(x, torch.Tensor) else str(x)[:160]
           for x in trial])
print("\n=== CACHED CURRENT BASE / PRIOR REFERENCES ===")
for nm in ["rh_baseview", "rh_vg", "rh_hidden", "rh_nf"]:
    ob = globals()[nm]
    print(nm, tuple(ob.shape), float(ob.mean()), float(ob.std()))
print("Current edits:", edited())
```
