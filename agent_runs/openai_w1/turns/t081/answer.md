The fresh exports use a different row order from the cached metadata, so the latest calibration is invalid. I’ll align rows by their exact posterior vectors and check the preceding full states as well. This avoids guessing the ordering; only agreement on both states will justify reusing the cached context.

```python
# Match cached posterior vectors to the actual exported rows, independently of metadata order.
r73_lookup = {}
for j, z in enumerate(r72_x[:, 960:]):
    r73_lookup.setdefault(z.tobytes(), []).append(j)
r73_matches = [r73_lookup.get(z.tobytes(), [])
               for z in r36_latent_target.numpy()]
r73_found = np.array([bool(v) for v in r73_matches])
print("=== VECTOR-BASED ALIGNMENT ===")
print("exact matches", int(r73_found.sum()), "/", len(r73_found),
      "multiple matches", sum(len(v) > 1 for v in r73_matches))

if r73_found.all():
    r73_raw = np.array([v[0] for v in r73_matches])
    r73_prev_error = np.max(
        np.abs(r36_previous.numpy() - r72_x[r73_raw-1]), axis=1)
    print("preceding-state max-error quantiles",
          np.quantile(r73_prev_error, [0, .5, .95, 1]))
    print("cached stay-id agreement",
          np.mean(np.asarray(r36_sid) == r72_ids[r73_raw]))
    for i in np.asarray(rr58_mi)[:4]:
        j = r73_raw[i]
        print("cache row", int(i), "actual identity",
              int(r72_ids[j]), float(r72_hours[j]),
              "cached", int(r36_sid[i]), float(r36_hour[i]))

    r73_measurements, r73_rows = [], []
    for i, j in enumerate(r73_raw):
        hour = float(r72_hours[j])
        if abs(hour-round(hour)) > 1e-6:
            continue
        rows = r36_world.get((int(r72_ids[j]), round(hour, 4)), [])
        rows = [row for row in rows if row.event not in ("insulin", "norepinephrine")]
        if any(row.event in ("glucose_finger", "glucose_lab",
                             "map_arterial", "map_cuff") for row in rows):
            r73_measurements.append(i)
            r73_rows.append(rows)
    pick = np.random.default_rng(7301).choice(
        len(r73_measurements), min(1200, len(r73_measurements)), replace=False)
    r73_mi = np.asarray(r73_measurements)[pick]
    r73_mrows = [r73_rows[j] for j in pick]
    r73_e0 = r71_observations(r73_mrows)
    diff = r73_e0 - r36_e[r73_mi]
    r73_z0 = r72_posterior(r73_e0, r73_mi)
    print("\n=== VECTOR-ALIGNED OBSERVATIONS ===")
    print("eligible", len(r73_measurements),
          "encoding RMSE", float(diff.square().mean().sqrt()),
          "posterior RMSE",
          float((r73_z0-r36_latent_target[r73_mi]).square().mean().sqrt()))
    r73_reading_dirs = {}
    for label, dg, dm, scale in (("glucose", 8., 0., 16.),
                                 ("map", 0., 2., 4.)):
        ep = r71_observations(r73_mrows, dg, dm)
        em = r71_observations(r73_mrows, -dg, -dm)
        dz = (r72_posterior(ep, r73_mi)-r72_posterior(em, r73_mi))/scale
        r73_reading_dirs[label] = dz.mean(0).numpy()
        print(label, "direction norm", float(dz.mean(0).norm()))
else:
    print("Cached targets cannot be aligned exactly; do not use either recent calibration.")

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 73\n"
            "Run 72 used incompatible export/cache ordering; its directions are invalid and rejected. "
            "Aligning cached rows to exports by exact 32-dimensional posterior vectors and checking previous states. "
            "No weights changed. Counts: 73 Python runs, 6 forward passes.\n")
```