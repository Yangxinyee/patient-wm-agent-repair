The sampled atom cache is consistent, but the encoder returns a tensor directly, so the test stopped on an unpacking error. I’ll remove that assumption and compare atoms across the full value range with the validated helper. Exact atoms with inaccurate posteriors would narrow the fault to pooling, context, or normalization.

```python
print("=== HELPER CONSTANTS ===")
for name in ("r79_norm", "r79_mlp", "rr_norm", "rr_mlp", "rr_atom"):
    fn = globals()[name]
    print(name, inspect.signature(fn), "constants=", fn.__code__.co_consts)

print("\n=== FULL-RANGE ATOM COMPARISON ===")
rr_sensor_idx111 = np.flatnonzero([
    d["code"] in rr_sensors for d in rr_desc4
])
rr_sample_idx111 = rr_sensor_idx111[
    np.linspace(0, len(rr_sensor_idx111)-1, 400, dtype=int)
]
rr_old_atoms111 = torch.stack([
    r95_correct_atom(**rr_desc4[int(i)]) for i in rr_sample_idx111
])
rr_atom_err111 = (rr_old_atoms111-rr_atoms4[rr_sample_idx111]).abs().max(1).values.numpy()
print("max atom differences", np.quantile(rr_atom_err111, [0, .5, .9, 1]))
for j in np.argsort(rr_atom_err111)[-3:]:
    print(rr_desc4[int(rr_sample_idx111[j])], float(rr_atom_err111[j]))

print("\n=== RECOMPUTED POSTERIOR ===")
rr_test_e111 = rr_encode4(~rr_is_drug4 | rr_is_end4, "encode", False)
rr_test_u111 = rr_encode4(rr_is_drug4 & ~rr_is_end4, "encode_action", True)
rr_best_ctx111 = rr_context_options4[rr_best4[2]]
print("encoder output shapes", rr_test_e111.shape, rr_test_u111.shape)
print("best context versus current max difference",
      float((rr_best_ctx111-rr_ctx4).abs().max()))
rr_test_z111 = rr_mlp(torch.cat([rr_best_ctx111, rr_test_e111, rr_test_u111], 1),
                     "z_post", "gelu")[:, :32]
rr_test_err111 = (rr_test_z111-rr_zexport4).square().mean(1).sqrt().numpy()
for label, mask in (
    ("all valid", rr_valid4),
    ("reading", rr_valid4 & rr_has_reading4),
    ("no reading", rr_valid4 & ~rr_has_reading4),
):
    print(label, "n", int(mask.sum()),
          "RMSE quantiles", np.quantile(rr_test_err111[mask], [0, .5, .9, 1]))

print("\n=== ENCODER TAIL ===")
for ins in dis.get_instructions(rr_encode4):
    if ins.offset >= 450:
        print(ins.opname, ins.argrepr)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 111\nCorrecting tensor-only encoder return handling and comparing full-range atoms, normalization constants, and best-context posterior. No edits. Counts: 111 Python runs, 6 forward passes.\n")
```