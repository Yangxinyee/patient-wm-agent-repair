The local core matches saved outputs to about \(1.5\times10^{-5}\) RMS, which is sufficient for finite-difference sensitivity tests. The records support persistent insulin effects and faster-decaying MAP effects. I’ll verify the decoder’s column layout and head inputs, then find latent directions that move the intended readings while minimizing cross-effects.

```python
print("=== DOSE DECODER COLUMN AUDIT ===")
u48 = post_inputs_audit[:, 384:].numpy()
p48 = rr39_action_decoder.predict(u48)
print("actual rates:", rr39_rates[:10])
print("decoder outputs:", p48[:10])
print("augmented targets:", rr39_Y[:10])
print("D:", rr41_D[:10])
print("gate:", rr41_gate[:10])
print("center fit coefficient norms:",
      [(k, np.linalg.norm(v.coef_, axis=1).tolist())
       for k, v in rr41_center_models.items()])
print("\n=== SAMPLE AND HELPER LAYOUT ===")
for k in ["bundle_records_audit", "core_probe_indices", "dt_all_audit",
          "num_spec0", "code_id0", "rr43_facts", "rr43_numeric"]:
    v = globals().get(k)
    print(k, type(v).__name__, getattr(v, "shape", ""))
    if isinstance(v, dict):
        print(" ", list(v.items())[:3])
    elif isinstance(v, (list, tuple)):
        print(" ", v[:2])
    elif k == "rr43_facts":
        print(v.head(2).to_string(index=False))
    elif k == "rr43_numeric":
        print(v[:3])
    elif k == "core_probe_indices":
        print(v[:10])
print("archive keys:", rr30_archive.files)
print("value centre layout:", type(VOCAB["value_bin_centres"]).__name__)
if isinstance(VOCAB["value_bin_centres"], dict):
    print("centre keys:", list(VOCAB["value_bin_centres"])[:5])
else:
    print("centre shape:", np.shape(VOCAB["value_bin_centres"]))
for fn48 in ["lin0", "mlp0", "norm0"]:
    print(fn48, inspect.signature(globals()[fn48]))
print("gap tensors:", [(n, SHAPES[n]) for n in NAMES if n.startswith("gap.")])
print("probe bundle examples:",
      [bundle_records_audit[int(i)] for i in core_probe_indices[:3]])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 48: dose and sensitivity input audit\n"
            "- Local core out RMS 1.53e-5 supports finite-difference calibration.\n"
            "- Checking rate decoder column order and constructing sensor-specific latent sensitivity tests.\n"
            "- Python runs: 48; forward passes: 4; no tensors edited.\n")
```