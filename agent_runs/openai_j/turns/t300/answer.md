The likelihood helper averages four groups of four latent samples. The earlier discrepancy may therefore be a comparison of individual groups with their average, rather than a faulty likelihood. I’ll verify that, then check whether the hidden features can improve MAP dose sensitivity without changing glucose sensitivity or exhausting the factual-loss budget.

```python
r262_expected = torch.as_tensor(tr259["pred"]["nll"])
r262_reconstructed = check_nll259.mean(0)
print("Correctly averaged NLL max error:",
      float((r262_reconstructed - r262_expected).abs().max()))

print("\nHEAD TENSOR SHAPES:")
for r262_k, r262_v in weights259.items():
    print(r262_k, tuple(r262_v.shape))
print("Output tilt v259:", np.asarray(v259).round(4))
print("Existing tilt wm_v151:", np.asarray(wm_v151).round(4))
print("Hidden scale range:", float(scale259.min()), float(scale259.max()))

print("\nCURRENT CACHED FACTUAL COST:")
for r262_key in ["train", "tune"]:
    r262_pack = packs259[r262_key]
    r262_orig = np.asarray(r262_pack["mc"]["nll"].mean(0))
    r262_delta = r262_pack["pred"]["nll"] - r262_orig
    if r262_key == "train":
        for r262_label, r262_mask in [("fit", fit259), ("heldout", cv259)]:
            print(r262_label, float(r262_delta[r262_mask].mean()))
    else:
        print("tuning", float(r262_delta.mean()))

print("\nDOSE MOMENT ROWS:")
for r262_i, r262_label in enumerate(labels259):
    print(r262_i, r262_label, slopes259[r262_label],
          "row_norm", round(float(np.linalg.norm(rows259[r262_i])), 5))

print("\nRELEVANT LATE HELPERS:")
r262_fns = [k for k, v in list(globals().items())
            if inspect.isfunction(v) and not k.startswith("r26")]
print(r262_fns[-40:])

print("\nCAUSAL NOTE EXCERPTS:")
for r262_item in chosen257[:5]:
    print(str(r262_item)[:700])

with open("/work/notes.md", "a") as r262_f:
    r262_f.write(
        "\n## Run 262: corrected likelihood audit\n"
        f"- Correctly averaged reconstruction max error: "
        f"{float((r262_reconstructed-r262_expected).abs().max()):.8g}.\n"
        "- Inspecting event-specific moment directions and factual budget. "
        "No weights changed.\n"
    )
```