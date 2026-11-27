All 5,215 reconstructed bundles match the exported atom counts. The representative baseline rollout is still weak: insulin doubling shifts glucose by −1.77, while norepinephrine doubling shifts arterial MAP by only +0.05. I’ll now check posterior reconstruction; near-numerical agreement will validate the encoding, whereas sizeable errors will halt calibration.

```python
rr_n4 = len(rr_keys4)
rr_x4 = torch.from_numpy(rr_state4["x"]).float()
rr_valid4 = np.r_[False, rr_state4["stay_id"][1:] == rr_state4["stay_id"][:-1]]
rr_prev4 = np.maximum(np.arange(rr_n4) - 1, 0)
rr_prevout4 = rr_x4[rr_prev4, :192]
rr_prevh4 = rr_x4[rr_prev4, 192:960].reshape(-1, 4, 192)
rr_zexport4 = rr_x4[:, 960:992]

rr_enc1_4 = {
    prefix: rr_mlp(rr_atoms4, prefix + ".mlp1")
    for prefix in ["encode", "encode_action"]
}

def rr_encode4(mask, prefix, empty_zero):
    mask = torch.as_tensor(mask, dtype=torch.bool)
    owner = rr_owner4[mask]
    count = torch.bincount(owner, minlength=rr_n4).float()
    summed = torch.zeros(rr_n4, 192)
    summed.index_add_(0, owner, rr_enc1_4[prefix][mask])
    pooled = summed / count.clamp_min(1).sqrt()[:, None]
    pooled = pooled + rr_linear(torch.log1p(count)[:, None], prefix + ".size")
    result = rr_norm(rr_mlp(pooled, prefix + ".mlp2"), prefix + ".norm")
    if empty_zero:
        result[count == 0] = 0
    return result

rr_routes4 = {
    "all_drug_actions": (~rr_is_drug4, rr_is_drug4),
    "drug_END_observation": (~rr_is_drug4 | rr_is_end4,
                             rr_is_drug4 & ~rr_is_end4),
    "drug_END_ignored": (~rr_is_drug4, rr_is_drug4 & ~rr_is_end4),
    "drug_END_both": (~rr_is_drug4 | rr_is_end4, rr_is_drug4),
}
rr_enc_options4 = {}
for route, (emask, umask) in rr_routes4.items():
    u = rr_encode4(umask, "encode_action", True)
    for ezero in [False, True]:
        rr_enc_options4[(route, ezero)] = (
            rr_encode4(emask, "encode", ezero), u
        )

rr_context_options4 = {}
for clock in ["csv_hours", "rounded_minutes"]:
    hours = (rr_state4["hour"].astype(np.float64) if clock == "csv_hours"
             else np.array([key[1] / 60 for key in rr_keys4]))
    dt = torch.tensor(np.maximum((hours - hours[rr_prev4]) * 3600, 0),
                      dtype=torch.float32)
    hflow = rr_prevh4 * torch.exp(
        -dt[:, None, None] * rr_original["core.flow.a"].exp()[None, None, :]
    )
    base = rr_prevout4 + rr_norm(hflow.mean(1), "core.out_norm")
    for logkind in ["log", "log1p"]:
        gapinput = (dt.clamp_min(1e-6).log() if logkind == "log"
                    else torch.log1p(dt))[:, None] / 10
        for activation in ["gelu", "silu"]:
            rr_context_options4[(clock, logkind, activation)] = (
                base + rr_mlp(gapinput, "gap", activation)
            )

rr_post_tests4 = []
for enc_key, (e, u) in rr_enc_options4.items():
    for ctx_key, ctx in rr_context_options4.items():
        hidden = F.gelu(rr_linear(torch.cat([ctx, e, u], 1), "z_post.0"))
        z = rr_linear(hidden, "z_post.2")[:, :32]
        rmse = (z - rr_zexport4).square().mean(1).sqrt().numpy()
        rr_post_tests4.append((float(rmse[rr_valid4].mean()),
                               enc_key, ctx_key))
rr_post_tests4.sort(key=lambda row: row[0])
print("=== BEST POSTERIOR RECONSTRUCTIONS ===")
for result in rr_post_tests4[:6]:
    print(result)

rr_best4 = rr_post_tests4[0]
rr_e4, rr_u4 = rr_enc_options4[rr_best4[1]]
rr_ctx4 = rr_context_options4[rr_best4[2]]
rr_input4 = torch.cat([rr_ctx4, rr_e4, rr_u4], 1)
rr_hidden4 = F.gelu(rr_linear(rr_input4, "z_post.0"))
rr_zlocal4 = rr_linear(rr_hidden4, "z_post.2")[:, :32]
rr_zerror4 = (rr_zlocal4 - rr_zexport4).square().mean(1).sqrt().numpy()
rr_has_reading4 = np.array([
    any(a["code"] in rr_sensors and not a["end"] for a in bundle)
    for bundle in rr_bundles4
])
rr_has_active4 = np.array([
    any(a["code"] in rr_drugs and not a["end"] for a in bundle)
    for bundle in rr_bundles4
])
for label, mask in [
    ("all", rr_valid4), ("readings", rr_valid4 & rr_has_reading4),
    ("active drugs", rr_valid4 & rr_has_active4),
    ("END/other", rr_valid4 & ~rr_has_reading4 & ~rr_has_active4)
]:
    print(label, "n", int(mask.sum()),
          "RMSE median/90th/max",
          np.quantile(rr_zerror4[mask], [.5, .9, 1]).tolist())

with Path("/work/notes.md").open("a") as f:
    f.write(
        f"\n## Run 103\nPosterior reconstruction best: {rr_best4!r}. "
        f"Valid-row maximum posterior RMSE: {rr_zerror4[rr_valid4].max():.8g}. "
        "Fresh records and exported state rows have matching atom counts. "
        "No weights edited.\nCounts: 103 Python runs, 6 forward passes.\n"
    )
```
