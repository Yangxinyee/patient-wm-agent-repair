The inventory revealed a separate reconstruction configuration, but the output was too large to show its validation results. I’ll use that configuration for a controlled latent-sensitivity test and report the agreement explicitly. Strong, separable glucose and MAP responses would support a posterior edit; weak responses would rule out simple posterior amplification.

```python
print("Decoder validation results:", decoder_checks109)
print("Recurrent RMSE:",
      float((h109 - sx_repair[core_rows104, 192:960].reshape(-1,4,192)).square().mean().sqrt()),
      float((out109 - sx_repair[core_rows104, :192]).square().mean().sqrt()))
probe_head110 = linked_head108.loc[
    linked_head108.event.isin(["glucose_finger", "map_arterial"])
].copy()
print("Probe readings:", len(probe_head110), probe_head110.event.value_counts().to_dict())
print("Posterior mean SD:", repair_z101.std(0).numpy().round(3))

archive110 = S
state_array110 = sx_repair.numpy().copy()
S = {"x": state_array110}
state_array110[core_rows104, :192] = out109.numpy()
state_array110[core_rows104, 192:960] = h109.reshape(-1,768).numpy()
base_head110 = local_value_head(probe_head110, repair_head_cfg,
                               zmode="prior_mc", nsamples=32, seed=917)
print("Precise-config mean RMSE:",
      float(np.sqrt(np.mean((np.asarray(base_head110["mean"]) -
                            probe_head110.predicted_mean.to_numpy())**2))))
latent_J110 = np.zeros((len(probe_head110), 32), dtype=np.float64)
latent_out_J110 = np.zeros((32, len(core_rows104), 192), dtype=np.float32)
eps110 = 0.25
try:
    for dim110 in range(32):
        zplus110 = repair_z101[core_pick104].clone()
        zminus110 = zplus110.clone()
        zplus110[:, dim110] += eps110
        zminus110[:, dim110] -= eps110
        hp110, op110 = step104(repair_hflow101[core_pick104],
                             repair_e101[core_pick104], repair_u101[core_pick104],
                             zplus110, mem109, heads=4, age_mode="both")
        hm110, om110 = step104(repair_hflow101[core_pick104],
                             repair_e101[core_pick104], repair_u101[core_pick104],
                             zminus110, mem109, heads=4, age_mode="both")
        means110 = []
        for hh110, oo110 in [(hp110, op110), (hm110, om110)]:
            state_array110[core_rows104, :192] = oo110.numpy()
            state_array110[core_rows104, 192:960] = hh110.reshape(-1,768).numpy()
            rr110 = local_value_head(probe_head110, repair_head_cfg,
                                    zmode="prior_mc", nsamples=32, seed=917)
            means110.append(np.asarray(rr110["mean"]))
        latent_J110[:, dim110] = (means110[0] - means110[1]) / (2*eps110)
        latent_out_J110[dim110] = ((op110-om110)/(2*eps110)).numpy()
finally:
    S = archive110
latent_summaries110 = {}
for ev110 in ["glucose_finger", "map_arterial"]:
    sel110 = probe_head110.event.eq(ev110).to_numpy()
    mean_J110 = latent_J110[sel110].mean(0)
    latent_summaries110[ev110] = mean_J110
    print(ev110, "mean-response vector norm:", float(np.linalg.norm(mean_J110)),
          "individual norm median:", float(np.median(np.linalg.norm(latent_J110[sel110], axis=1))),
          "largest coordinates:", sorted(enumerate(mean_J110), key=lambda kv: -abs(kv[1]))[:6])
print("Response-vector cosine:",
      float(np.dot(*latent_summaries110.values()) /
            np.prod([np.linalg.norm(vv110) for vv110 in latent_summaries110.values()])))
with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 105–110: local intervention reconstruction\n"
            "Recovered exact-head configuration repair_head_cfg and 4-head recurrent step. "
            "Measured finite-difference posterior-z sensitivity at 32 factual drug bundles; "
            "results are in latent_J110 and latent_out_J110. No edits installed.\n")
```
