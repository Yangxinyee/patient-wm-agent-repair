The action embedding matches exactly, but the core reconstruction does not: its summary is close while its four memory slots differ substantially. I won’t fit an edit from those unverified contrasts. I’ll check whether the discrepancy is a centering or scaling convention, and recompute the posterior error only on valid drug bundles.

```python
r116_rows = r115_test_rows
r116_true_h = SX[r116_rows, 192:960].reshape(-1, 4, 192)
r116_pred_h = r115_h
r116_diff_h = r116_pred_h - r116_true_h
print("=== CORE DISCREPANCY ===")
print("True/pred slot mean, SD:",
      float(r116_true_h.mean()), float(r116_pred_h.mean()),
      float(r116_true_h.std()), float(r116_pred_h.std()))
print("Mean difference across channels RMS:",
      float(np.sqrt(np.mean(r116_diff_h.mean(-1) ** 2))))
print("Channel-centered difference RMS:",
      float(np.sqrt(np.mean((r116_diff_h - r116_diff_h.mean(-1, keepdims=True)) ** 2))))
r116_true_centered = r116_true_h - r116_true_h.mean(-1, keepdims=True)
r116_pred_centered = r116_pred_h - r116_pred_h.mean(-1, keepdims=True)
r116_scale = np.sum(r116_true_centered * r116_pred_centered, axis=-1) / np.maximum(
    np.sum(r116_pred_centered ** 2, axis=-1), 1e-12)
print("Best per-slot scale mean/SD:", r116_scale.mean(), r116_scale.std())
print("Centered/rescaled RMSE:", float(np.sqrt(np.mean(
    (r116_true_centered - r116_scale[..., None] * r116_pred_centered) ** 2))))
r116_good = r114_drug_rows
print("Finite posterior rows:", np.isfinite(r115_z_model[r116_good].numpy()).all(1).sum(), len(r116_good))
print("Valid drug posterior RMSE:", float(torch.sqrt(torch.mean(
    (r115_z_model[r116_good] - post_truth[r116_good]) ** 2))))
print("=== CORE HELPER OPERATIONS ===")
for nm in ["repair_core_step", "repair_simple_block", "repair_self_attention"]:
    fu = globals()[nm]
    print("\n", nm, "names:", fu.__code__.co_names)
    print("constants:", [c for c in fu.__code__.co_consts if isinstance(c, (str, int, float))])
    if nm == "repair_core_step":
        for ins in dis.get_instructions(fu):
            if ins.opname in ["LOAD_CONST", "LOAD_GLOBAL", "LOAD_METHOD", "BINARY_OP", "STORE_FAST", "RETURN_VALUE"]:
                print(ins.opname, ins.argrepr)
print("\n=== CORE TENSOR SHAPES ===")
print([(n, SHAPES[n]) for n in NAMES if any(s in n for s in
       ["coda", "c_out", "b_in", "a_in", "delta_in", "prelude_norm"])])
print("\n=== NOTEBOOK CORE AUDIT EXCERPTS ===")
for line in open("/work/notes.md"):
    if any(s in line.lower() for s in ["core reconstruction", "coda", "inputboth", "slot", "core-step", "core step"]):
        print(line.strip()[:600])
```
