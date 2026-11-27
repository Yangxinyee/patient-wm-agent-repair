The corrected factual features and paired-dose contrasts are cached, so we can fit without repeating the core reconstruction. I’ll check their alignment and tilt convention first. A saved-weight equivalence test must reproduce the fitted logits before any candidate goes to the GPU.

```python
import torch, dis

def r121_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

print("=== METADATA ===")
print(head_meta.columns.tolist())
print(head_meta.head(3).to_string(index=False))
print("\n=== VERIFIED FACTUAL FEATURES ===")
for r121_k, r121_v in verified_head.items():
    r121_a = r121_np(r121_v)
    print(r121_k, r121_a.shape, r121_a.dtype)
print("numeric inputs, first rows:\n", r121_np(verified_head["numbers"])[:4])

print("\n=== PAIRED CONTRAST FEATURES ===")
for r121_key, r121_d in r117_cf.items():
    print(r121_key, {
        k: (r121_np(v).shape, str(r121_np(v).dtype))
        for k, v in r121_d.items()
    })
print("\nProbability match:", np.max(np.abs(
    r117_base_probs - r121_np(verified_head["probs"])
)))
print("Recorded bin fields:",
      [c for c in head_meta.columns if "bin" in c or "value" in c])

print("\n=== SMALL HELPER IMPLEMENTATIONS ===")
for r121_name in ["corrected_register_head", "repair_tilt", "repair_centres17"]:
    print("\n", r121_name)
    dis.dis(globals()[r121_name])

print("\n=== WEIGHT CACHE CONVENTION ===")
print("wt references:", wt.__code__.co_names)
print("Original head shapes:",
      SHAPES["attr_heads.value_bin.2.weight"],
      SHAPES["attr_heads.value_bin.2.bias"])
print("Active edits:", edited())
```
