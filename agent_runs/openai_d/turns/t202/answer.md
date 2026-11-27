The norepinephrine correction is a gated, paired-neuron output tilt, so I can test amplification without changing the state dynamics. The diagnostic stopped on an indexing error; no weights changed. I’ll use the existing loss helper and verify that it reproduces the cached loss before trusting the amplification screen.

```python
print("===== LOSS HELPER REFERENCES =====")
print("r138_nll names:", r138_nll.__code__.co_names)
print("_target_index type:", type(_target_index).__name__)
if isinstance(_target_index, tuple):
    print("index component shapes:",
          [getattr(x, "shape", None) for x in _target_index])

r175_logits = torch.as_tensor(r170_local_logits).numpy().astype(np.float64)
r175_g = np.asarray(r165_ne_G, dtype=np.float64)
r175_d = np.asarray(r165_ne_D, dtype=np.float64)
r175_pair = ((r175_g + r175_d) * ndtr(r175_g + r175_d)
             - (r175_g - r175_d) * ndtr(r175_g - r175_d))
r175_out = torch.as_tensor(
    r171_combined_backup["attr_heads.value_bin.2.weight"]
).numpy()[:, r170_rows[0]].astype(np.float64)

r175_base_loss = np.asarray(r138_nll(r175_logits))
print("Cached loss reconstruction maximum error:",
      float(np.max(np.abs(r175_base_loss - np.asarray(r170_local_loss)))))

print("\n===== LOCAL AMPLIFICATION SCREEN =====")
r175_screen = []
for s in [1.0, 1.25, 1.5, 1.8]:
    logits = r175_logits + (s - 1) * r175_pair[..., None] * r175_out
    loss = np.asarray(r138_nll(logits))
    delta = loss - np.asarray(r170_orig_loss)
    row = {"NE_multiplier": s, "all": float(delta.mean()),
           "fit": float(delta[np.asarray(r137_fit, dtype=bool)].mean()),
           "validation": float(delta[np.asarray(r137_val, dtype=bool)].mean()),
           "increment": float((loss - r175_base_loss).mean())}
    r175_screen.append(row)
    print(row)

print("\n===== ACTIVE NE FEATURE DISTRIBUTION =====")
mask = np.asarray(r165_ne_active, dtype=bool)
for label, x in [("gate", r175_g), ("direction", r175_d),
                 ("paired activation", r175_pair)]:
    print(label, np.quantile(x[:, mask], [.01, .1, .5, .9, .99]))

print("\n===== RECORD TARGET STRUCTURE =====")
for drug, value in record_targets.items():
    print(drug, type(value).__name__)
    if isinstance(value, dict):
        for key, item in value.items():
            if hasattr(item, "shape"):
                print(" ", key, tuple(item.shape),
                      repr(item)[:150] if np.prod(item.shape) <= 24 else "")
            else:
                print(" ", key, repr(item)[:250])
    else:
        print(repr(value)[:500])

print("\n===== PANEL AND METADATA VARIABLES =====")
for n, v in list(globals().items()):
    if isinstance(v, pd.DataFrame) and (
        "panel" in n.lower() or "meta" in n.lower()
    ):
        print(n, v.shape, v.columns.tolist()[:14])
```
