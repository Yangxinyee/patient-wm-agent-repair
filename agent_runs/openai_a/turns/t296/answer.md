Norepinephrine initiation changes sharply near recorded MAP 65, while the displayed prior and next-pressure means look smoother. That is a candidate identification boundary, not yet evidence of a null effect. I’ll test local dose and outcome jumps across bandwidths, with history adjustment and a prior-pressure placebo. Strong first stages and stable outcome estimates would support a conclusion; weak stages or placebo jumps would undermine it.

```python
from scipy.linalg import lstsq

ne_rd_frame = p_ne.copy()
carry_map = panel["map_arterial"].groupby(level="stay_id").ffill()
carry_cuff = panel["map_cuff"].groupby(level="stay_id").ffill()
for lag in (1, 2, 3, 4):
    ne_rd_frame[f"hist_map_{lag}"] = (
        carry_map.groupby(level="stay_id").shift(lag).reindex(ne_rd_frame.index)
    )
ne_rd_frame["hist_cuff_1"] = (
    carry_cuff.groupby(level="stay_id").shift(1).reindex(ne_rd_frame.index)
)

def ne_threshold_rd(bandwidth, degree=1, initiation=False, adjusted=True):
    frame = ne_rd_frame
    x0 = frame["map_arterial"].to_numpy(dtype=float) - 65.0
    valid = (
        np.isfinite(x0)
        & (np.abs(x0) <= bandwidth)
        & (x0 != 0)
        & frame["map_next"].notna().to_numpy()
        & frame["ne_prev"].notna().to_numpy()
    )
    if initiation:
        valid &= frame["ne_prev"].to_numpy() == 0
    f = frame.iloc[np.flatnonzero(valid)]
    x = x0[valid] / bandwidth
    z = (x < 0).astype(float)

    columns = [np.ones(len(f))]
    for power in range(1, degree + 1):
        columns.extend([x**power, z * x**power])
    if adjusted:
        control_names = [
            "ne_prev", "hist_map_1", "hist_map_2", "hist_map_3",
            "hist_map_4", "hist_cuff_1", "map_cuff"
        ]
        for name in control_names:
            v = f[name].to_numpy(dtype=float)
            missing = ~np.isfinite(v)
            if missing.all():
                continue
            v[missing] = np.nanmedian(v)
            sd = v.std()
            if sd > 1e-8:
                columns.append((v - v.mean()) / sd)
            if missing.any() and not missing.all():
                columns.append(missing.astype(float))
        columns.append(f.index.get_level_values("t").to_numpy(dtype=float) / 24)
    W = np.column_stack(columns)

    d = f["norepinephrine"].to_numpy(dtype=float)
    y = f["map_next"].to_numpy(dtype=float)
    placebo = f["hist_map_1"].to_numpy(dtype=float)
    placebo[~np.isfinite(placebo)] = np.nanmedian(placebo)

    joint = np.column_stack([z, d, y, placebo])
    residual = joint - W @ lstsq(W, joint, lapack_driver="gelsy")[0]
    zr, dr, yr, pr = residual.T
    denominator = float(zr @ dr)
    zz = float(zr @ zr)
    beta_local = float(zr @ yr) / denominator
    groups = pd.factorize(f.index.get_level_values("stay_id"))[0]

    def clustered_se(error, denom):
        sums = np.bincount(groups, weights=zr * error)
        return float(np.sqrt(np.dot(sums, sums)) / abs(denom))

    first_stage = denominator / zz
    fs_se = clustered_se(dr - first_stage * zr, zz)
    reduced_form = float(zr @ yr) / zz
    reduced_se = clustered_se(yr - reduced_form * zr, zz)
    prior_jump = float(zr @ pr) / zz
    return {
        "bw": bandwidth, "degree": degree, "init": initiation,
        "adjusted": adjusted, "n": len(f),
        "dose_jump": first_stage, "fs_t": first_stage / fs_se,
        "outcome_jump": reduced_form, "outcome_se": reduced_se,
        "beta": beta_local,
        "beta_se": clustered_se(yr - beta_local * dr, denominator),
        "prior_jump": prior_jump,
    }

ne_rd_results = pd.DataFrame([
    ne_threshold_rd(bw, degree, initiation, adjusted)
    for initiation in (False, True)
    for adjusted in (False, True)
    for degree in (1, 2)
    for bw in (4, 6, 8)
])
print(ne_rd_results.round(4).to_string(index=False))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## NE policy-boundary analysis; Python 254, forwards 32/50\n"
        "- Recorded initiation probability changes sharply near arterial MAP 65.\n"
        "- Testing local IV at the empirical boundary, excluding the rounded value 65, "
        "with cluster SEs, bandwidth/degree checks and history adjustment.\n"
        "- No NE effect accepted and no weights changed by this analysis.\n"
    )
```