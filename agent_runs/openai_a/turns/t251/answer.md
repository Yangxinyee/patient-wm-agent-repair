Finger-glucose inputs also overlap, so I won’t use a hard off-manifold gate. The validated direction remains more reliable than the ridge-1 direction. I’ll try a modest scale increase from 6 to 7, preserving the fallback. This trial is worthwhile only if the GPU doubling response improves and factual NLL remains within the limit.

```python
# Verify that the recovered helper reproduces the accepted fallback exactly.
recreated_fallback = make_selective_contrast_graft(
    w_fallback, b_fallback, 6.0, gate_strength=30.0
)
max_recreation_error = max(
    float(np.max(np.abs(as_np_recovery(stronger6_checkpoint[n])
                        - recreated_fallback[n])))
    for n in sc_names
)
print("Fallback recreation max error:", max_recreation_error)
assert max_recreation_error < 1e-6, "Helper globals do not match the fallback."

# Audit the whitened regression, without changing its fitted direction.
print("\nWHITENED REGRESSION AUDIT:")
for lam in (0.1, 0.3, 1.0):
    v = np.linalg.solve(g3 + lam * np.eye(g3.shape[0]), h3)
    w_check = r3_Q @ t3 @ v
    ref = r3_candidates[(lam, 1.0)]
    print(lam, "weight error:", np.max(np.abs(w_check - ref["w"])),
          "bias error:", abs(float(-sc_mean @ w_check) - ref["b"]))

scale7_trial = make_selective_contrast_graft(
    w_fallback, b_fallback, 7.0, gate_strength=30.0
)
for n, tensor in scale7_trial.items():
    save_edit(n, torch.from_numpy(np.asarray(tensor, dtype=np.float32).copy()))
scale7_checkpoint = {
    n: torch.as_tensor(load(n)).detach().clone() for n in sc_names
}
print("\nInstalled selective lambda=0.3, scale=7 trial.")

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Modest scale-7 trial; Python 219, forwards 23/50\n"
        "- Finger-only factual and synthetic decoder distributions overlap; "
        "a hard off-manifold gate is not justified.\n"
        "- Exact helper/fallback equality verified before changing weights.\n"
        "- Installed lambda=0.3, scale=7, gate strength=30 through the same "
        "four head tensors. stronger6_checkpoint remains protected.\n"
        "- Trial is unaccepted pending GPU factual and dose-edit validation.\n"
    )
```
