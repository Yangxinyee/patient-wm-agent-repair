The saved comparisons now align readings by stay, hour and sensor, so the remaining error is not simply a row mismatch. I’ll test the two gap conventions alongside the leading register layouts. If none reproduces the external predictions, I’ll reject this as the sole explanation.

```python
print("probe columns:", list(probe104.columns))
print("example numeric inputs:", numeric104[:2], "z values:", zval104[:4])

def assay_head110(gap_scale, value_column, flag_column, activation, type_index=1):
    old_gap = mlp0(torch.log(dt104[:, None]) / 10, "gap")
    new_gap = mlp0(torch.log(dt104[:, None]) * gap_scale, "gap")
    view = ctx104 - old_gap + new_gap
    query = norm0(view[:, None, :] + zh104 +
                  W0["type_emb.weight"][type_index], "query_norm")
    numeric = np.zeros((len(ix104), 8), dtype=np.float32)
    numeric[:, 0] = has104
    numeric[:, 1] = age105
    numeric[:, value_column] = zval104
    if flag_column is not None:
        numeric[:, flag_column] = has104
    with_code = (query + codeemb104[:, None, :] + binemb104[:, None, :] +
                 W0["embed.bins.rate.weight"][0] + W0["last_status.weight"][0] +
                 lin0(torch.tensor(numeric), "last_numeric")[:, None, :])
    hidden = act0(F.linear(with_code, W0["attr_heads.value_bin.0.weight"],
                           W0["attr_heads.value_bin.0.bias"]), activation)
    logits = F.linear(hidden, W0["attr_heads.value_bin.2.weight"],
                      W0["attr_heads.value_bin.2.bias"]) + trans104[:, None, :]
    probability = torch.softmax(logits, -1).mean(1)
    means = (probability * centres104).sum(-1).numpy()
    nlls = -torch.log(probability[torch.arange(len(label104)), label104]
                      .clamp_min(1e-12)).numpy()
    return probability, means, nlls

results110 = []
for gs110 in (0.1, 1.0, 0.0):
    for vc110 in (2, 6):
        for fc110 in (3, None):
            for act110 in ("gelu", "relu"):
                pp110, mm110, nn110 = assay_head110(gs110, vc110, fc110, act110)
                results110.append({
                    "gap": gs110, "value_col": vc110, "flag_col": fc110,
                    "activation": act110,
                    "nll": float(nn110.mean()),
                    "mean_mae": float(np.abs(mm110 - probe104.predicted_mean).mean()),
                    "nll_mae": float(np.abs(nn110 - probe104.nll).mean())
                })
rank110 = sorted(results110, key=lambda r: r["nll_mae"])
print("\n=== GAP / REGISTER FRONTIER ===")
for r110 in rank110[:12]:
    print(r110)

best110 = rank110[0]
p110, mean110, nll110 = assay_head110(
    best110["gap"], best110["value_col"], best110["flag_col"],
    best110["activation"])
comparison110 = probe104.copy()
comparison110["mean_error_new"] = mean110 - comparison110.predicted_mean
comparison110["nll_error_new"] = nll110 - comparison110.nll
print("\n=== BEST BY SENSOR ===")
print(comparison110.groupby("event").agg(
    mean_bias=("mean_error_new", "mean"),
    mean_mae=("mean_error_new", lambda x: np.abs(x).mean()),
    nll_bias=("nll_error_new", "mean"),
    nll_mae=("nll_error_new", lambda x: np.abs(x).mean())).to_string())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 110: explicit head-gap assay\n"
            f"- Best matched-reading convention: {best110}\n"
            "- Tested preprocessing only; no weights edited.\n"
            "- Python runs: 110; forward passes: 4.\n")
```