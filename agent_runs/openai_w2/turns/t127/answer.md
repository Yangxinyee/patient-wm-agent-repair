The posterior perturbations barely changed either prediction, which rules out simple latent amplification as a useful repair. The recurrent reconstruction is accurate, while the decoder still has sampling-related disagreement. I’ll now extract features from the prelude’s direct action pathway and test whether they can separate dose changes from factual variation.

```python
def prelude_features111(h, e, u, z):
    prefix111 = "core.prelude.0"
    tokens111 = torch.cat([
        h + Wrepair["core.slots"],
        e[:, None, :], u[:, None, :],
        lin104(z, "z_in")[:, None, :]
    ], dim=1)
    normed111 = norm104(tokens111, prefix111 + ".norm1")
    attended111 = attention104(normed111, normed111,
                              prefix111 + ".attn", 4)
    before_ff111 = tokens111 + attended111
    hidden111 = F104.gelu(lin104(norm104(before_ff111, prefix111 + ".norm3"),
                                prefix111 + ".ffn.0"))
    after_ff111 = before_ff111 + lin104(hidden111, prefix111 + ".ffn.2")
    E111 = norm104(after_ff111[:, :4], "core.prelude_norm")
    return hidden111[:, :4], E111, attended111[:, :4]

test_hidden111, test_E111, _ = prelude_features111(
    repair_hflow101[core_pick104], repair_e101[core_pick104],
    repair_u101[core_pick104], repair_z101[core_pick104])
tokens_test111 = torch.cat([
    repair_hflow101[core_pick104] + Wrepair["core.slots"],
    repair_e101[core_pick104, None, :],
    repair_u101[core_pick104, None, :],
    lin104(repair_z101[core_pick104], "z_in")[:, None, :]
], dim=1)
reference_E111 = norm104(layer104(tokens_test111, "core.prelude.0", 4)[:, :4],
                        "core.prelude_norm")
print("Prelude reconstruction max error:",
      float((test_E111-reference_E111).abs().max()))

standard111 = (repair_input101.numpy()-input_center103)/input_scale103
if np.max(np.abs(input_X103[:, 1:]-standard111)) < 1e-4:
    design_slice111 = slice(1, None)
elif np.max(np.abs(input_X103[:, :-1]-standard111)) < 1e-4:
    design_slice111 = slice(None, -1)
else:
    raise ValueError("Input contrast layout is not recognized.")

prelude_hidden111 = np.empty((len(repair_z101),4,768), dtype=np.float32)
prelude_E111 = np.empty((len(repair_z101),4,192), dtype=np.float32)
prelude_attended111 = np.empty_like(prelude_E111)
prelude_cf_hidden111 = {}
prelude_cf_E111 = {}
for key111 in [None] + list(input_D103):
    if key111 is None:
        u_all111 = repair_u101
    else:
        raw_delta111 = input_D103[key111][:, design_slice111] * input_scale103
        u_all111 = repair_u101 + torch.as_tensor(raw_delta111[:, -192:],
                                                dtype=torch.float32)
        prelude_cf_hidden111[key111] = np.empty_like(prelude_hidden111)
        prelude_cf_E111[key111] = np.empty_like(prelude_E111)
    for start111 in range(0, len(repair_z101), 128):
        sl111 = slice(start111, start111+128)
        hh111, ee111, aa111 = prelude_features111(
            repair_hflow101[sl111], repair_e101[sl111],
            u_all111[sl111], repair_z101[sl111])
        if key111 is None:
            prelude_hidden111[sl111] = hh111.numpy()
            prelude_E111[sl111] = ee111.numpy()
            prelude_attended111[sl111] = aa111.numpy()
        else:
            prelude_cf_hidden111[key111][sl111] = hh111.numpy()
            prelude_cf_E111[key111][sl111] = ee111.numpy()
    print("Extracted", key111, "features")
print("Factual train/test rows:", int(residual_train102.sum()), int(residual_test102.sum()))
print("Dose ranges:", np.nanmin(residual_rates102,axis=0),
      np.nanmax(residual_rates102,axis=0))
print("Contrast target shapes:", {k: np.shape(v) for k,v in residual_targets102.items()})
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 110 result and run 111\n"
            "Posterior-z perturbations are essentially ignored by the recurrent model "
            "(mean response norms about 0.00057 glucose and 0.000026 MAP per z-unit). "
            "Reject posterior amplification. Extracted factual and dose-probe prelude "
            "FFN hidden features and E inputs for an analytical direct-state edit. "
            "No edits installed.\n")
```
