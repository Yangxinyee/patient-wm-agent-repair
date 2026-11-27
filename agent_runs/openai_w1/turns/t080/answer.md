GELU reconstructs drug-bundle posteriors closely, but the observation calibration is using mismatched bundles: some purported measurement bundles appear to be five minutes later. I’ll check against the raw state exports and rebuild the calibration using exact times. If correctly timed observations match the cached encoding, we can proceed without changing the encoder implementation.

```python
r72_exports = []
for k in (2, 4):
    with np.load(f"/work/fwd_{k}/states_original.npz") as a:
        r72_exports.append({key: a[key].copy() for key in a.files})
r72_ids = np.concatenate([a["stay_id"] for a in r72_exports])
r72_hours = np.concatenate([a["hour"] for a in r72_exports])
r72_x = np.concatenate([a["x"] for a in r72_exports])
r72_idx = np.asarray(r36_indices, dtype=int)

print("=== RAW EXPORT ALIGNMENT ===")
print("r33 ids/hours equal exports:",
      np.array_equal(r33_ids, r72_ids), np.array_equal(r33_hours, r72_hours))
print("target z max error:",
      np.max(np.abs(r36_latent_target.numpy() - r72_x[r72_idx, 960:])))
print("previous-state max error:",
      np.max(np.abs(r36_previous.numpy() - r72_x[r72_idx-1])))
for i in np.asarray(rr58_mi)[:4]:
    s = r72_idx[i]
    print("row", int(i), "raw", int(r72_ids[s]), float(r72_hours[s]),
          "cached hour", float(r36_hour[i]))

r72_measurements, r72_rows = [], []
for i, s in enumerate(r72_idx):
    hour = float(r72_hours[s])
    if abs(hour - round(hour)) > 1e-6:
        continue
    rows = r36_world.get((int(r72_ids[s]), round(hour, 4)), [])
    rows = [row for row in rows if row.event not in ("insulin", "norepinephrine")]
    if any(row.event in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff")
           for row in rows):
        r72_measurements.append(i)
        r72_rows.append(rows)
r72_rng = np.random.default_rng(7201)
r72_pick = r72_rng.choice(len(r72_measurements),
                         min(1200, len(r72_measurements)), replace=False)
r72_mi = np.asarray(r72_measurements)[r72_pick]
r72_mrows = [r72_rows[j] for j in r72_pick]
r72_e0 = r71_observations(r72_mrows)
r72_diff = r72_e0 - r36_e[r72_mi]
print("\n=== EXACT-TIME OBSERVATIONS ===")
print("eligible bundles", len(r72_measurements),
      "encoding RMSE/max", float(r72_diff.square().mean().sqrt()),
      float(r72_diff.abs().max()))

def r72_posterior(e, ix):
    inp = torch.cat((r36_ctx[ix], e, r71_u_all[ix]), dim=1)
    return r33_linear(r33_act(r33_linear(inp, "z_post.0"), "gelu"),
                      "z_post.2")[:, :32]

r72_z0 = r72_posterior(r72_e0, r72_mi)
print("exact-time posterior RMSE:",
      float((r72_z0-r36_latent_target[r72_mi]).square().mean().sqrt()))
r72_reading_dirs = {}
for label, dg, dm, scale in (("glucose", 8.0, 0.0, 16.0),
                             ("map", 0.0, 2.0, 4.0)):
    ep = r71_observations(r72_mrows, dg, dm)
    em = r71_observations(r72_mrows, -dg, -dm)
    dz = (r72_posterior(ep, r72_mi)-r72_posterior(em, r72_mi))/scale
    r72_reading_dirs[label] = dz.mean(0).numpy()
    print(label, "direction norm", float(dz.mean(0).norm()),
          "individual derivative RMS", float(dz.square().mean().sqrt()))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 72\n"
            "Run 71 found GELU drug posterior reconstruction mean error 0.003869, "
            "but observation calibration mixed integer reading times with fractional action bundles. "
            "Checking raw exported metadata and recalibrating using exact integer bundle times. "
            "No weights edited. Counts: 72 Python runs, 6 forward passes.\n")
```