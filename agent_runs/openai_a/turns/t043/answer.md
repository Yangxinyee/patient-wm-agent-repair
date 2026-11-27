Reducing the last-value scale helped slightly, but the remaining error is too large to trust the CPU head. I’ll test the value and presence-flag positions in the eight-field register. The correct layout should bring the CPU NLL close to 1.6346; if none does, I’ll investigate the world-view calculation.

```python
layout_results = []
lw = W["last_numeric.weight"]
add_without_value_flag = (
    head_parts_tune["additive"]
    - last_z_tensor[:, None] * lw[:, 2][None, :]
    - has_last_tensor[:, None] * lw[:, 3][None, :]
)
for value_col in range(2, 8):
    for flag_col in [-1] + list(range(2, 8)):
        if flag_col == value_col:
            continue
        for last_status in [0, 1]:
            additive = add_without_value_flag + last_z_tensor[:, None] * lw[:, value_col]
            if flag_col >= 0:
                additive = additive + has_last_tensor[:, None] * lw[:, flag_col]
            if last_status:
                additive = additive + W["last_status.weight"][1] - W["last_status.weight"][0]
            hidden = F.gelu(lin(head_q + additive, "attr_heads.value_bin.0"))
            logits = lin(hidden, "attr_heads.value_bin.2") + head_parts_tune["transition"]
            probs = logits.softmax(-1)
            means = (probs * head_centres).sum(-1).numpy()
            nll = -probs[torch.arange(len(probs)), head_obsbins].clamp_min(1e-20).log().numpy()
            layout_results.append({
                "value_col": value_col, "flag_col": flag_col, "status": last_status,
                "rmse": float(np.sqrt(np.mean((means - head_gpu_mean)**2))),
                "nll": float(nll.mean())
            })
print("=== REGISTER LAYOUT TEST ===")
print(pd.DataFrame(layout_results).sort_values("rmse").head(12).to_string(index=False))
print("\n=== INITIAL CPU ADDITIVE CHECK ===")
reg_check = torch.zeros(len(reference_meta), 8)
reg_check[:, 0] = has_last_tensor
reg_check[:, 1] = torch.tensor(
    np.log1p(reference_meta.last_age_seconds.to_numpy()) / 10, dtype=torch.float32)
reg_check[:, 2] = last_z_tensor
reg_check[:, 3] = has_last_tensor
manual_additive = (
    lin(reg_check, "last_numeric")
    + W["embed.code.weight"][torch.tensor(reference_meta.code.to_numpy())]
    + W["embed.bins.value.weight"][torch.tensor(reference_meta.last_bin.to_numpy())]
    + W["embed.bins.rate.weight"][0] + W["last_status.weight"][0]
)
print("max additive reconstruction difference:",
      float((manual_additive - head_parts_tune["additive"]).abs().max()))
print("\n=== REACHED-HOUR ORIGINAL READOUTS ===")
print(dose_baseline[dose_baseline.reached.astype(bool)].groupby("edit")["shift"].mean().to_string())
with open("/work/notes.md", "a") as f:
    f.write("\n## Register layout test; Python 38, forward 4\n")
    f.write(json.dumps(sorted(layout_results, key=lambda x: x["rmse"])[:5]) + "\n")
    f.write("- No edits. Unweighted all-hour dose CSV means dilute the evaluator readouts; filter reached hours and match stay weighting.\n")
```
