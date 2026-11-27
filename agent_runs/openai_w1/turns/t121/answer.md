The caches already match fresh GELU calculations, so that hypothesis was not supported. The run then stopped because atom descriptors are dictionaries, not tuples. I’ll correct that inspection and finish the posterior test; this will distinguish an atom-input problem from a later encoder problem.

```python
print("=== ATOM DESCRIPTORS ===")
print(rr_desc4[:3])
print("rr_encode4 local names", rr_encode4.__code__.co_varnames)

rr_check_idx110 = np.flatnonzero(~rr_is_drug4)[:200]
rr_check110 = torch.stack([
    rr_atom(**rr_desc4[int(i)]) if isinstance(rr_desc4[int(i)], dict)
    else rr_atom(*rr_desc4[int(i)])
    for i in rr_check_idx110
])
print("measurement atom cache max error",
      float((rr_check110-rr_atoms4[rr_check_idx110]).abs().max()))
print("best previous configuration", rr_best4)

print("\n=== RECOMPUTED POSTERIOR ===")
rr_test_e110, rr_test_ec110 = rr_encode4(~rr_is_drug4 | rr_is_end4, "encode", False)
rr_test_u110, rr_test_uc110 = rr_encode4(rr_is_drug4 & ~rr_is_end4, "encode_action", True)
rr_test_z110 = rr_mlp(torch.cat([rr_ctx4, rr_test_e110, rr_test_u110], 1),
                     "z_post", "gelu")[:, :32]
rr_test_err110 = (rr_test_z110-rr_zexport4).square().mean(1).sqrt().numpy()
for label, mask in (
    ("all valid", rr_valid4),
    ("reading", rr_valid4 & rr_has_reading4),
    ("no reading", rr_valid4 & ~rr_has_reading4),
):
    print(label, "n", int(mask.sum()),
          "RMSE quantiles", np.quantile(rr_test_err110[mask], [0, .5, .9, 1]))

print("\n=== ENCODER TAIL ===")
for ins in dis.get_instructions(rr_encode4):
    if ins.offset >= 450:
        print(ins.opname, ins.argrepr)

print("\n=== ORIGINAL AND FRESH NORMALIZATION CONSTANTS ===")
for name in ("r79_norm", "rr_norm", "rr_mlp"):
    fn = globals()[name]
    print(name, inspect.signature(fn), fn.__code__.co_consts)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 110\nPer-atom caches match fresh GELU exactly. Run 109 stopped on dictionary-valued descriptors; completing atom and posterior checks with correct descriptor handling. No edits. Counts: 110 Python runs, 6 forward passes.\n")
```