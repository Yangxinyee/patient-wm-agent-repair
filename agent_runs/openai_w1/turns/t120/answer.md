Both encoders use the same pooling formula. The discrepancy may therefore be in the cached atom encodings rather than the formula itself. I’ll compare those caches with fresh GELU and SiLU calculations, then recompute the posterior without changing weights. Numerical agreement would confirm the cache correction.

```python
print("=== CACHE CONSISTENCY ===")
for prefix in ("encode", "encode_action"):
    cached = rr_enc1_4[prefix]
    for act in ("gelu", "silu"):
        fresh = rr_mlp(rr_atoms4, prefix + ".mlp1", act)
        print(prefix, act, "cached RMSE", float((cached-fresh).square().mean().sqrt()),
              "max", float((cached-fresh).abs().max()))

print("\n=== ATOM CACHE CONSISTENCY ===")
rr_check_idx109 = np.flatnonzero(~rr_is_drug4)[:100]
rr_check109 = torch.stack([rr_atom(*rr_desc4[int(i)]) for i in rr_check_idx109])
print("measurement cache max error",
      float((rr_check109-rr_atoms4[rr_check_idx109]).abs().max()))
print("first descriptors", rr_desc4[:4])
print("best previous configuration", rr_best4)

print("\n=== FRESH GELU ENCODING TEST ===")
rr_saved_enc109 = rr_enc1_4
rr_enc1_4 = {
    prefix: rr_mlp(rr_atoms4, prefix + ".mlp1", "gelu")
    for prefix in ("encode", "encode_action")
}
rr_test_e109, rr_test_ec109 = rr_encode4(~rr_is_drug4 | rr_is_end4, "encode", False)
rr_test_u109, rr_test_uc109 = rr_encode4(rr_is_drug4 & ~rr_is_end4, "encode_action", True)
rr_test_z109 = rr_mlp(torch.cat([rr_ctx4, rr_test_e109, rr_test_u109], 1),
                     "z_post", "gelu")[:, :32]
rr_test_err109 = (rr_test_z109-rr_zexport4).square().mean(1).sqrt().numpy()
for label, mask in (
    ("all valid", rr_valid4),
    ("reading", rr_valid4 & rr_has_reading4),
    ("no reading", rr_valid4 & ~rr_has_reading4),
):
    print(label, "n", int(mask.sum()),
          "RMSE quantiles", np.quantile(rr_test_err109[mask], [0, .5, .9, 1]))
rr_enc1_4 = rr_saved_enc109

print("\n=== ENCODER TAIL ===")
for ins in dis.get_instructions(rr_encode4):
    if ins.offset >= 450:
        print(ins.opname, ins.argrepr)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 109\nComparing cached per-atom encodings against fresh nonlinearities and testing a clean posterior reconstruction. Temporary local arrays only; no weights edited. Counts: 109 Python runs, 6 forward passes.\n")
```