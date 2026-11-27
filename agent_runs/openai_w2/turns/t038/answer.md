The held-out decoder performs poorly: gaining appreciable probe sensitivity requires substantial changes on factual features. That argues against a head-only repair. I’ll instead examine an action-path edit with factual recalibration, and check within-stay persistence so the desired rollout effects are grounded in the records rather than an arbitrary multiplier.

```python
print("=== ACTION-PATH TENSORS ===")
print([(n, SHAPES[n]) for n in NAMES if n.startswith("encode_action.")])
print("\n=== ORIGINAL FORWARD SUMMARIES ===")
for j in (1, 2, 4):
    summary_j = json.loads(Path(f"/work/fwd_{j}/summary.json").read_text())
    print(j, json.dumps(summary_j)[:1400])

original_dose_table = pd.read_csv("/work/fwd_1/dose_edit_original.csv")
print("Dose-table columns:", list(original_dose_table.columns))
if "stay_id" in original_dose_table:
    benchmark_stays = [int(x) for x in original_dose_table["stay_id"].unique()]
else:
    benchmark_stays = [int(x) for x in pd.read_csv("/data/tuning.csv")["stay_id"].unique()[:100]]
print("Benchmark tuning stays:", benchmark_stays[:100])
benchmark_stays = benchmark_stays[:100]

print("\n=== DIFFERENCED, MEASUREMENT-ERROR IV PERSISTENCE ===")
print("F index:", F.index.names)
iv_persistence_results = []
for gold, proxy, drug, drug_effects in [
    ("glucose_finger", "glucose_lab", "insulin", (-1.55, -1.8, -2.0)),
    ("glucose_lab", "glucose_finger", "insulin", (-1.55, -1.8, -2.0)),
    ("map_arterial", "map_cuff", "norepinephrine", (3.0, 3.6, 4.0)),
    ("map_cuff", "map_arterial", "norepinephrine", (3.0, 3.6, 4.0)),
]:
    raw_gold = F[gold + "_raw"]
    raw_proxy = F[proxy + "_raw"]
    gold_prev = raw_gold.groupby(level=0, sort=False).shift(1)
    gold_next = raw_gold.groupby(level=0, sort=False).shift(-1)
    drug_now = F[drug + "_current"]
    drug_prev = drug_now.groupby(level=0, sort=False).shift(1)
    for instrument_name, instrument in [
        ("other_sensor_lag1", raw_proxy.groupby(level=0, sort=False).shift(1)),
        ("same_sensor_lag2", raw_gold.groupby(level=0, sort=False).shift(2)),
    ]:
        valid_iv = (
            F["hour"].between(3, 23) & raw_gold.notna()
            & gold_prev.notna() & gold_next.notna() & instrument.notna()
            & drug_now.notna() & drug_prev.notna())
        positions_iv = np.flatnonzero(valid_iv.to_numpy())
        if len(positions_iv) < 500:
            continue
        init_iv = F[gold + "_initial"].iloc[positions_iv].to_numpy(float)
        init_missing_iv = ~np.isfinite(init_iv)
        init_iv = np.where(init_missing_iv, np.nanmedian(init_iv), init_iv)
        time_iv = F["hour"].iloc[positions_iv].to_numpy(float)
        controls_iv = np.column_stack([
            np.ones(len(positions_iv)), time_iv, time_iv**2,
            init_iv, init_missing_iv.astype(float)])
        moments_iv = np.column_stack([
            (raw_gold - gold_prev).iloc[positions_iv].to_numpy(float),
            (gold_next - raw_gold).iloc[positions_iv].to_numpy(float),
            (drug_now - drug_prev).iloc[positions_iv].to_numpy(float),
            instrument.iloc[positions_iv].to_numpy(float)])
        residual_iv = moments_iv - controls_iv @ np.linalg.lstsq(
            controls_iv, moments_iv, rcond=None)[0]
        xi, dyi, ddi, zi = residual_iv.T
        denominator_iv = float(zi @ xi)
        cluster_ids_iv = F.index.get_level_values(0).to_numpy()[positions_iv]
        _, cluster_codes_iv = np.unique(cluster_ids_iv, return_inverse=True)
        for assumed_b_iv in drug_effects:
            yi = dyi - assumed_b_iv * ddi
            rho_iv = float((zi @ yi) / denominator_iv)
            scores_iv = zi * (yi - rho_iv * xi)
            cluster_scores_iv = np.bincount(cluster_codes_iv, weights=scores_iv)
            se_iv = float(np.sqrt(np.sum(cluster_scores_iv**2)) / abs(denominator_iv))
            iv_persistence_results.append({
                "sensor": gold, "instrument": instrument_name, "n": len(xi),
                "b": assumed_b_iv, "rho": rho_iv, "cluster_se": se_iv,
                "instrument_x_cov": float(np.mean(zi * xi)),
            })

iv_persistence_table = pd.DataFrame(iv_persistence_results)
print(iv_persistence_table.round(4).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 33: head-only route rejected; within-stay persistence check\n"
            "Held-out factual-zero decoders cannot recover much probe response: at weight 512 "
            "insulin gain 0.112 costs factual RMS 1.427 rate units; NE gain 0.280 costs 0.873. "
            "Do not install decoder edits. Next route: amplify action pathway and recalibrate "
            "factual predictions using code/last-value transition weights. "
            "Differenced IV removes stay intercepts; cross-sensor lag1 and same-sensor lag2 "
            "instruments check persistence conditional on externally supported drug coefficients. "
            "Interpret with missing-reading selection and instrument-strength caution.\n"
            + iv_persistence_table.to_json(orient="records") + "\n"
            + "benchmark_stays=" + repr(benchmark_stays) + "\n")
```