The action encoding and cached states are now identified. I’ll change only insulin’s rate and dose, recompute the posterior and recurrent step, and measure the resulting state contrasts. If those contrasts exceed the baseline reconstruction error and vary consistently with dose, they may be useful for fitting a correction.

```python
dose_probe_states = {}
dose_probe_actions = {}
dose_probe_posts = {}
saved_rm_rates = np.array(rm_rates, copy=True)
saved_rm_doses = np.array(rm_doses, copy=True)

try:
    probe_ctx, _ = rec_context(rm_prev, rm_dt)
    for multiplier in (0.5, 1.0, 2.0):
        rm_rates = saved_rm_rates * multiplier
        rm_doses = saved_rm_doses * multiplier
        atoms = rm_insulin_atoms(3, 2, "left", True, "gelu")
        action = rm_set(atoms, "encode_action", activation="gelu")
        posterior = rm_mlp(
            torch.cat([probe_ctx, rm_empty_e, action], dim=1), "z_post"
        )[:, :32]
        result = r131_step(
            rm_prev, rm_dt, rm_empty_e, action, posterior,
            r140_mem, r140_age_cached, r140_mask, age_mode="both_input"
        )
        components = list(result) if isinstance(result, (tuple, list)) else [result]
        slots = next(v for v in components if v.ndim == 3)
        out = next(v for v in components if v.ndim == 2 and v.shape[1] == 192)
        dose_probe_states[multiplier] = torch.cat(
            [out, slots.reshape(len(out), -1), posterior], dim=1
        ).detach().cpu().numpy()
        dose_probe_actions[multiplier] = action.detach().cpu().numpy()
        dose_probe_posts[multiplier] = posterior.detach().cpu().numpy()
finally:
    rm_rates = saved_rm_rates
    rm_doses = saved_rm_doses

probe_truth = np.asarray(rm_states["x"])[rm_pure_idx]
def probe_rms(a):
    return float(np.sqrt(np.mean(np.asarray(a, dtype=np.float64) ** 2)))

print("=== RECONSTRUCTION ERROR AND DOSE CONTRASTS ===")
for label, delta in [
    ("unchanged reconstruction error", dose_probe_states[1.0] - probe_truth),
    ("half-dose contrast", dose_probe_states[0.5] - dose_probe_states[1.0]),
    ("double-dose contrast", dose_probe_states[2.0] - dose_probe_states[1.0]),
]:
    print(label, {
        "out_RMS": probe_rms(delta[:, :192]),
        "slots_RMS": probe_rms(delta[:, 192:960]),
        "posterior_RMS": probe_rms(delta[:, 960:]),
    })
print("contrast-direction cosine:",
      float(np.sum((dose_probe_states[2.0][:, :192] - dose_probe_states[1.0][:, :192]) *
                   (dose_probe_states[0.5][:, :192] - dose_probe_states[1.0][:, :192])) /
            (np.linalg.norm(dose_probe_states[2.0][:, :192] - dose_probe_states[1.0][:, :192]) *
             np.linalg.norm(dose_probe_states[0.5][:, :192] - dose_probe_states[1.0][:, :192]))))

print("\n=== MATCHED NEXT-READING METADATA ===")
print("cached stays:", np.unique(rm_states["stay_id"])[:3])
for table_name in ("head_meta_tune", "head_meta_train", "head_meta_extra"):
    table = globals()[table_name]
    matched = table[table["state_row"].isin(rm_pure_idx)]
    print(table_name, len(matched), "stays:", matched["stay_id"].unique()[:3])

print("\n=== PROBABILITY-HELPER OUTPUT CONTRACT ===")
print(cpu_head_probs.__code__.co_names)
print(repr(cpu_head_probs.__code__.co_consts)[-1800:])
```
