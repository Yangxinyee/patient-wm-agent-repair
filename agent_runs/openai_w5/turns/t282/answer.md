The recovered audit shows that the dose channel is roughly proportional to recorded exposure, but the centering term varies substantially across sensors. That makes centering worth testing, though it is not yet evidence of leakage. I will inspect the installed equations next; a center built from post-dose features could absorb interventions, whereas a pre-dose-only center should not do so immediately.

```python
print("=== CHANNEL AND HEAD DESIGN NOTES ===")
print(r246_notes[14279:20561])

print("\n=== RESERVED HEAD MAPPING ===")
print("_pair_for_control:", globals().get("_pair_for_control"))
print("private_pairs:", globals().get("private_pairs"))
print("_gate_table:\n", globals().get("_gate_table"))

r247_w0 = np.asarray(load("attr_heads.value_bin.0.weight"))
r247_b0 = np.asarray(load("attr_heads.value_bin.0.bias"))
r247_dw0 = r247_w0 - np.asarray(load_original("attr_heads.value_bin.0.weight"))
r247_db0 = r247_b0 - np.asarray(load_original("attr_heads.value_bin.0.bias"))
r247_top = np.argsort(np.linalg.norm(r247_dw0, axis=1))[-12:][::-1]
print("\n=== LARGEST FIRST-LAYER EDITS ===")
for r247_j in r247_top:
    r247_row = r247_w0[r247_j]
    print(int(r247_j), "bias", round(float(r247_b0[r247_j]), 5),
          "delta_norm", round(float(np.linalg.norm(r247_dw0[r247_j])), 5),
          "private", np.round(r247_row[[21, 61, 15, 66]], 5).tolist(),
          "largest_inputs", np.argsort(np.abs(r247_row))[-6:].tolist())

print("\n=== OTHER AVAILABLE OBJECT NAMES ===")
print(" ".join(k for k in sorted(list(globals()))
               if not k.startswith(("r245_", "r246_", "r247_", "__"))
               and k not in ("np", "torch", "Path", "inspect"))[:1500])
```
