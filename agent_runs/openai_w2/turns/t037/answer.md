The frozen configuration exactly reproduces the saved features, with near-zero channel biases. The corrected probe changes are much smaller than the invalid comparison suggested. I’ll now test whether a dose-response correction can generalize across held-out stays and latent draws while remaining nearly silent on factual records. Failure would argue against a head-only repair.

```python
probe_seeds = list(range(1701, 1709))
probe_draw_store = {k: [] for k in ("h0", "h1", "p0", "p1", "m0", "m1")}
saved_S_for_draws = local_value_head.__globals__["S"]
try:
    for seed_probe in probe_seeds:
        local_value_head.__globals__["S"] = S_reference
        draw_original = local_value_head(
            head_data, repair_head_cfg, bin_side="left", zmode="prior_mc",
            nsamples=1, return_features=True, seed=seed_probe)
        local_value_head.__globals__["S"] = S_probe
        draw_probe = local_value_head(
            head_data, repair_head_cfg, bin_side="left", zmode="prior_mc",
            nsamples=1, return_features=True, seed=seed_probe)
        for tag, result in (("0", draw_original), ("1", draw_probe)):
            for short, full in (("h", "hidden"), ("p", "prob"), ("m", "mean")):
                probe_draw_store[short + tag].append(
                    np.asarray(result[full], dtype=np.float32))
finally:
    local_value_head.__globals__["S"] = saved_S_for_draws
probe_draw_store = {k: np.stack(v) for k, v in probe_draw_store.items()}

print("Correctly matched mean probe shifts:")
probe_shift_table = head_data[["event"]].copy()
probe_shift_table["shift"] = (
    probe_draw_store["m1"] - probe_draw_store["m0"]).mean(axis=0)
print(probe_shift_table.groupby("event")["shift"].agg(["mean", "std"]).to_string())
print("Exposure summary:", head_exposures.describe().to_dict())
print("Centre-table keys:", list(VOCAB["value_bin_centres"].keys())[:15])

decoder_h0 = probe_draw_store["h0"].reshape(-1, 192).astype(np.float64)
decoder_dh = (probe_draw_store["h1"] - probe_draw_store["h0"]).reshape(-1, 192).astype(np.float64)
decoder_center = decoder_h0.mean(axis=0)
decoder_scale = np.maximum(decoder_h0.std(axis=0), 0.01)
decoder_B0 = np.column_stack([
    (decoder_h0 - decoder_center) / decoder_scale,
    np.ones(len(decoder_h0))])
decoder_D = np.column_stack([
    decoder_dh / decoder_scale, np.zeros(len(decoder_dh))])

decoder_target_rows = np.zeros((len(head_data), 2))
decoder_glucose = head_data["event"].str.startswith("glucose").to_numpy()
decoder_map = head_data["event"].str.startswith("map").to_numpy()
decoder_target_rows[:, 0] = (
    decoder_glucose * (head_exposures["insulin"].to_numpy() > 0) * 0.5)
decoder_target_rows[:, 1] = (
    decoder_map * (head_exposures["norepinephrine"].to_numpy() > 0) * 0.3)
decoder_targets = np.tile(decoder_target_rows, (len(probe_seeds), 1))

decoder_stays = head_data["stay_id"].unique().copy()
np.random.default_rng(1873).shuffle(decoder_stays)
decoder_train_rows = head_data["stay_id"].isin(decoder_stays[:70]).to_numpy()
decoder_train = np.tile(decoder_train_rows, len(probe_seeds))
decoder_test = ~decoder_train

Btrain = decoder_B0[decoder_train]
Dtrain = decoder_D[decoder_train]
Ttrain = decoder_targets[decoder_train]
C_factual = Btrain.T @ Btrain / len(Btrain)
C_probe = Dtrain.T @ Dtrain / len(Dtrain)
R_probe = Dtrain.T @ Ttrain / len(Dtrain)
decoder_reg = np.eye(193) * 1e-4
decoder_reg[-1, -1] = 1e-8

from scipy.linalg import eigh
probe_ratios = np.sqrt(np.maximum(
    eigh(C_probe, C_factual + decoder_reg, eigvals_only=True), 0))
print("Largest probe/factual feature ratios:", probe_ratios[-10:])

decoder_fits = {}
decoder_diagnostics = []
for weight_probe in (0.5, 2, 8, 32, 128, 512):
    w_decoder = np.linalg.solve(
        C_factual + weight_probe * C_probe + decoder_reg,
        weight_probe * R_probe)
    decoder_fits[weight_probe] = w_decoder
    row = {"probe_weight": weight_probe}
    for j, drug_decoder in enumerate(("insulin", "norepinephrine")):
        pred_factual = decoder_B0[decoder_test] @ w_decoder[:, j]
        pred_probe = decoder_D[decoder_test] @ w_decoder[:, j]
        truth_probe = decoder_targets[decoder_test, j]
        row[drug_decoder + "_factual_rms"] = float(np.sqrt(np.mean(pred_factual**2)))
        row[drug_decoder + "_probe_gain"] = float(
            np.dot(pred_probe, truth_probe) / max(np.dot(truth_probe, truth_probe), 1e-12))
        row[drug_decoder + "_probe_rmse"] = float(
            np.sqrt(np.mean((pred_probe - truth_probe)**2)))
    decoder_diagnostics.append(row)
decoder_diagnostics = pd.DataFrame(decoder_diagnostics)
print(decoder_diagnostics.to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 32: multi-draw, held-out dose-direction diagnostic\n"
            "Frozen head reconstruction exactly matches saved original features. "
            "Collected 8 matched single-prior draws (1701..1708) in probe_draw_store. "
            "70/30 stay split, standardized hidden features plus intercept. "
            "Fitted small linear dose-residual decoders using factual-zero and probe-response "
            "constraints; no weight edits installed.\n"
            + decoder_diagnostics.to_json(orient="records") + "\n")
```