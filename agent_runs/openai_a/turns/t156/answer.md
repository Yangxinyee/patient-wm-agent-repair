Adding ages after projection did not resolve the mismatch. Also, that test used a working memory array rather than the best cached configuration, so its errors are not directly comparable to the earlier result. I’ll compare configurations consistently and test minute rounding and empty-world memory entries. A sharp reduction in error would identify the missing convention.

```python
r140_mem, r140_age_cached, r140_mask = r131_memory_options[
    (True, 3600.0, 1.0, "both_input")
]
print("Target slot RMS:", float((rec_h_target ** 2).mean().sqrt()))
print("First time gaps:", rm_dt[:6].reshape(-1).tolist())

r140_js = []
r140_ages_raw = torch.zeros((*r140_mask.shape, 1), dtype=torch.float32)
r140_ages_round = torch.zeros_like(r140_ages_raw)
r140_mem_zero = r140_mem.clone()
all_ix = np.arange(len(rm_states["x"]))
hours64 = np.asarray(rm_states["hour"], dtype=np.float64)

for j, ix in enumerate(rm_pure_idx):
    js = np.flatnonzero(
        (rm_states["stay_id"] == rm_states["stay_id"][ix]) & (all_ix <= ix)
    )[-128:]
    r140_js.append(js)
    n = len(js)
    r140_ages_raw[j, :n, 0] = torch.tensor(
        (hours64[ix] - hours64[js]) * 3600, dtype=torch.float32
    )
    r140_ages_round[j, :n, 0] = torch.tensor(
        (np.round(hours64[ix] * 60) - np.round(hours64[js] * 60)) * 60,
        dtype=torch.float32
    )
    empty_positions = np.flatnonzero(as_numpy(rec_counts[js]) == 0)
    r140_mem_zero[j, empty_positions] = 0

r140_age_raw = rm_mlp(torch.log1p(r140_ages_raw), "core.age")
r140_age_round = rm_mlp(torch.log1p(r140_ages_round), "core.age")
print("Rebuilt versus cached age RMSE:",
      float(((r140_age_raw - r140_age_cached) ** 2).mean().sqrt()))

r140_trials = {}
for label, mem, age, dt in [
    ("cached_best", r140_mem, r140_age_cached, rm_dt),
    ("rebuilt_raw", r140_mem, r140_age_raw, rm_dt),
    ("rounded_time", r140_mem, r140_age_round, (rm_dt / 60).round() * 60),
    ("zero_empty_memory", r140_mem_zero, r140_age_raw, rm_dt),
    ("zero_empty_rounded", r140_mem_zero, r140_age_round, (rm_dt / 60).round() * 60),
]:
    hp, op = r131_step(
        rm_prev, dt, rm_empty_e, r139_u, rm_target_z,
        mem, age, r140_mask, "both_input"
    )
    he = ((hp - rec_h_target) ** 2).mean(dim=(1, 2)).sqrt()
    oe = ((op - rec_out_target) ** 2).mean(dim=1).sqrt()
    r140_trials[label] = (hp.clone(), op.clone(), he.clone(), oe.clone())
    print(label, "slot mean/max", float(he.mean()), float(he.max()),
          "out mean/max", float(oe.mean()), float(oe.max()))
```
