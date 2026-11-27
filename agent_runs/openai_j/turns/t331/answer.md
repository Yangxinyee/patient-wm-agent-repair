The cached frontier suggests that insulin strength 0.86 and MAP gain 7 with the low-strength calibration can create a better factual-loss margin. I’ll install that candidate, keeping the current edit as a backup. Full tuning-set factual validation and paired rollouts will confirm or refute whether the safety gain is worth the modest reduction in response strength.

```python
key285 = "i0.86_g7_soft_low"
candidate285 = r265_candidates[key285]
backup285 = {
    name: np.asarray(load(name).detach().cpu() if isinstance(load(name), torch.Tensor)
                     else load(name), dtype=np.float32).copy()
    for name in edited()
}
np.savez("/work/calibrated_backup285.npz", **backup285)

assert set(candidate285) == {
    "attr_heads.value_bin.0.weight", "attr_heads.value_bin.0.bias",
    "attr_heads.value_bin.2.weight", "attr_heads.value_bin.2.bias"
}
candidate_arrays285 = {}
for name285, tensor285 in candidate285.items():
    array285 = (tensor285.detach().cpu().numpy() if isinstance(tensor285, torch.Tensor)
                else np.asarray(tensor285)).astype(np.float32, copy=True)
    assert tuple(array285.shape) == tuple(SHAPES[name285])
    assert np.isfinite(array285).all()
    candidate_arrays285[name285] = array285

for name285 in list(edited()):
    if name285 not in candidate_arrays285:
        drop_edit(name285)
for name285, array285 in candidate_arrays285.items():
    save_edit(name285, torch.as_tensor(array285))
np.savez("/work/safety_candidate285.npz", **candidate_arrays285)

print("Installed:", key285)
print("Edited tensors:", edited())
print(front283.loc[front283.key == key285].to_string(index=False))
print("Tuning IDs:", np.asarray(ids268).shape,
      int(np.asarray(ids268).min()), int(np.asarray(ids268).max()))
print("First validation batch:", [int(x) for x in np.asarray(ids268)[:100]])
print("Sequential IDs:", np.array_equal(
    np.asarray(ids268), np.arange(2020000000, 2020000400)))

print("=== EARLIER GLUCOSE CALIBRATION SCREEN ===")
tab285 = pd.read_csv("/work/glucose_calibration_frontier.csv")
print("Columns:", list(tab285.columns), "shape:", tab285.shape)
sort285 = next((c for c in ["cv_dNLL", "tune_dNLL", "cv_loss"] if c in tab285), None)
print((tab285.sort_values(sort285) if sort285 else tab285).head(5).to_string(index=False))

print("=== ACTION EXTRAPOLATION ROLLOUT COMPARISON ===")
for k285 in [35, 36]:
    d285 = pd.read_csv(f"/work/fwd_{k285}/dose_edit_edited.csv")
    means285 = d285.loc[d285.reached].groupby(["edit", "stay_id"]).shift.mean().groupby("edit").mean()
    print(k285, means285.to_dict())

with open("/work/notes.md", "a") as f285:
    f285.write(
        "\n## Run 285: safer calibrated candidate installed\n"
        "- Installed i0.86_g7_soft_low from the closed-form statistical candidate screen.\n"
        "- Previous gain-8.5 candidate preserved in calibrated_backup285.npz.\n"
        "- Only four value-head tensors edited; recurrent core and action encoder unchanged.\n"
        "- This trades a small response reduction for a larger factual safety margin.\n"
        "- GPU factual and paired rollout validation pending.\n"
        "- Counts: 285 Python runs, 38 forward passes.\n"
    )
```
