The recurrent input scale is not near zero, so that does not explain the weak response. I also found that the local attention layout and memory-age conventions were not explicitly fixed in the last test. I’ll compare those conventions against saved GPU states before trusting another sensitivity estimate. Low state-reconstruction error would support the chosen convention; large error would limit its use.

```python
print("CURRENT CONVENTIONS:", wm_kv_layout, wm_memory_mode)
print("PREVIOUS BEST:", best67)
print("PREVIOUS TRIALS:", recurrence_trials67[:3])

check_rows82 = np.asarray(sample80[::4], dtype=int)
trials82 = []
for kv82 in ("global", "perhead"):
    wm_kv_layout = kv82
    for mem82 in ("age_then_plainnorm", "norm_then_age", "age_then_norm"):
        wm_memory_mode = mem82
        for mode82, scale82 in (("input", 1.0), ("input", .1), ("key", 1.0), ("key", .1)):
            opt82 = dict(include_current=True, age_mode=mode82,
                         age_units=3600.0, age_scale=scale82)
            he82, oe82 = [], []
            for j82 in check_rows82:
                hh82 = local_step67(int(j82), s4["x"][j82:j82+1, 960:992], **opt82)
                actual_h82 = torch.as_tensor(s4["x"][j82:j82+1, 192:960]).reshape(1, 4, 192)
                he82.append(float(torch.sqrt(torch.mean((hh82 - actual_h82)**2))))
                oo82 = wm_norm(hh82.mean(1), "core.out_norm")
                oe82.append(float(torch.sqrt(torch.mean(
                    (oo82 - torch.as_tensor(s4["x"][j82:j82+1, :192]))**2))))
            trials82.append(dict(kv=kv82, memory=mem82, **opt82,
                                 h_rmse=np.mean(he82), out_rmse=np.mean(oe82)))
trials82.sort(key=lambda t: t["out_rmse"])
print(pd.DataFrame(trials82).head(10).to_string(index=False))

best82 = trials82[0]
wm_kv_layout, wm_memory_mode = best82["kv"], best82["memory"]
step_options82 = {k: best82[k] for k in
                  ("include_current", "age_mode", "age_units", "age_scale")}
print("SELECTED:", best82)

jac82 = np.full_like(jac80, np.nan)
for a82, j82 in enumerate(sample80):
    z082 = np.asarray(s4["x"][j82, 960:992], dtype=np.float32)
    zs82 = np.vstack([z082, z082[None, :] + .25*np.eye(32, dtype=np.float32)])
    hh82 = local_step67(int(j82), zs82, **step_options82)
    mm82, ev82 = next_means_crn74(int(j82), hh82, nsamples=12, seed=913)
    if mm82 is None:
        continue
    for d82, prefix82 in enumerate(("glucose", "map")):
        ii82 = np.flatnonzero(np.char.startswith(np.asarray(ev82, dtype=str), prefix82))
        if len(ii82):
            jac82[a82, d82] = (
                np.asarray(mm82)[1:, ii82].mean(axis=1)
                - np.asarray(mm82)[0, ii82].mean()) / .25
print("Revalidated latent sensitivity norm quantiles:",
      np.nanquantile(np.linalg.norm(jac82, axis=2), [.1, .5, .9], axis=0))
print("Saved h RMS across checked rows:",
      np.sqrt(np.mean(s4["x"][check_rows82, 192:960]**2)))
print("Current edits:", edited())
```