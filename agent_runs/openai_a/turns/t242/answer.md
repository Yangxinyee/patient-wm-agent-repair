Stronger regularization gives a larger doubling contrast at roughly the current candidate’s screened factual cost, though its halving response is uncertain. I’ll install that candidate reversibly. It will be worth retaining only if GPU rollouts improve and factual NLL stays within the limit.

```python
trial16_config = {
    "regularization": 1.0, "scale": 16.0, "gate_strength": 30.0
}
trial16_fit = r3_candidates[(1.0, 1.0)]
trial16_weights = make_selective_contrast_graft(
    trial16_fit["w"], trial16_fit["b"], 16.0, gate_strength=30.0
)

assert set(trial16_weights) == set(stronger6_checkpoint)
for name, tensor in trial16_weights.items():
    assert tuple(tensor.shape) == tuple(SHAPES[name])
    assert np.isfinite(tensor).all()
for name, tensor in trial16_weights.items():
    save_edit(name, torch.from_numpy(np.asarray(tensor, dtype=np.float32)))

print("Installed reversible candidate:", trial16_config)
print("Screen:")
print(stronger_screen_df.query(
    "`lambda` == 1 and penalty == 1 and scale == 16"
).round(6).to_string(index=False))
print("GPU tuning IDs:", audit_tuning_ids[:3], audit_tuning_ids[97:100])
print("Edited tensors:", edited())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Reversible ridge-1 trial; Python 212, forwards 21/50\n"
        "- Installed r3 lambda=1, scale=16, finger gate strength=30.\n"
        "- CPU local doubling contrast -3.970 versus current stronger6 -3.390 "
        "at comparable cached NLL cost; halving contrast is uncertain.\n"
        "- stronger6_checkpoint remains the GPU-tested fallback.\n"
        "- Candidate is NOT accepted pending paired GPU factual and rollout tests.\n"
    )
```
