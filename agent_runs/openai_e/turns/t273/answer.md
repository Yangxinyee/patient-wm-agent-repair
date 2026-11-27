The audit confirms that balancing helps the insulin half-dose response, but the strong balanced edit failed factual fit. Norepinephrine corrections also consume substantial headroom. I’ll try a smaller joint edit: 85% insulin strength, a 35% balanced blend, and 20% of the tested multidirection norepinephrine correction. It must pass factual validation before its rollout improvements count.

```python
base236 = np.load("/work/baseline77.npz")["delta"].astype(np.float64)
with np.load("/work/insulin_candidate63.npz") as a236:
    balanced236 = a236["delta"].astype(np.float64) * (1.6 / 1.9)
    print("Archived balanced strength:", float(a236["strength"]))

reconstructed219_236 = base236 + 0.15 * (balanced236 - base236)
saved219_236 = np.load("/work/insulin_trial219.npz")["delta"].astype(np.float64)
error236 = float(np.max(np.abs(reconstructed219_236 - saved219_236)))
print("Trial219 reconstruction max error:", error236)
assert error236 < 1e-4, "Candidate composition differs from the documented contract"

insulin_scale236, insulin_blend236, ne_scale236 = 0.85, 0.35, 0.20
insulin_delta236 = insulin_scale236 * (
    (1.0 - insulin_blend236) * base236 + insulin_blend236 * balanced236)
ne_delta236 = ne_scale236 * np.load(
    "/work/ne_candidate151.npz")["ne_delta"].astype(np.float64)
joint_delta236 = (insulin_delta236 + ne_delta236).astype(np.float32)

original236 = load_original("core.b_in.weight")
original_array236 = (original236.detach().cpu().numpy()
                     if isinstance(original236, torch.Tensor)
                     else np.asarray(original236))
new_weight236 = np.ascontiguousarray(original_array236 + joint_delta236, dtype=np.float32)
assert new_weight236.shape == original_array236.shape
assert np.isfinite(new_weight236).all()
save_edit("core.b_in.weight", torch.from_numpy(new_weight236))

np.savez("/work/joint236.npz",
         delta=joint_delta236,
         insulin_delta=insulin_delta236.astype(np.float32),
         ne_delta=ne_delta236.astype(np.float32),
         insulin_scale=insulin_scale236,
         insulin_blend=insulin_blend236,
         ne_scale=ne_scale236)
print("Installed unaccepted joint trial:",
      {"insulin_scale": insulin_scale236, "blend": insulin_blend236,
       "ne_scale": ne_scale236, "delta_norm": float(np.linalg.norm(joint_delta236))})
print("Edited tensors:", edited())

with Path("/work/notes.md").open("a") as f236:
    f236.write(
        "\n## Run 236: conservative joint repair trial\n"
        "Installed original b_in + .85*((1-.35)*baseline77 + "
        ".35*candidate63_delta*1.6/1.9) + .20*ne_candidate151_ne_delta. "
        f"Trial219 reconstruction max error={error236:.8g}. "
        "Saved exact composition in joint236.npz. Unaccepted pending paired factual "
        "validation on the fresh Forward27 stays and intervention validation. "
        "No gradients or weight optimizer. Python runs: 236; forwards: 27.\n"
    )
```