The probe aligns with the cached readings, and context reconstruction agrees to numerical precision. It produces measurable head-context changes, especially for MAP. That supports testing a correction, but not yet installing one. I’ll recover the latent draws and existing head construction so I can measure intervention sensitivity against factual variation; excessive factual variation would make this approach unsuitable.

```python
print("--- Relevant tensor dimensions ---")
for n in NAMES:
    if n.startswith(("attr_heads.value_bin", "last_numeric", "query_norm",
                     "z_head", "type_emb", "embed.numeric")):
        print(n, SHAPES[n])

print("\n--- Cached reconstruction and candidate objects ---")
suffixes = ("98", "99", "104", "105", "106")
r128_items = [(k, v) for k, v in list(globals().items())
              if k.endswith(suffixes) and not callable(v)]
for k, v in sorted(r128_items):
    desc = r126_desc(v)
    if isinstance(v, np.ndarray) and v.size <= 8:
        desc["values"] = v.tolist()
    print(k, desc)

print("\n--- Exact-input cache comparisons ---")
for name in ("wc_draws", "wc_alt_draws", "latent98", "x98", "x99",
             "input98", "inputs99"):
    if name in globals():
        v = globals()[name]
        if hasattr(v, "shape") and tuple(v.shape) == tuple(r124_x.shape):
            a = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
            print(name, "max abs difference",
                  float(np.max(np.abs(a - r124_x.numpy()))))

print("\n--- Further causal evidence ---")
print(r126_notes[3600:6300])
print("\n--- Kernel-related notebook excerpts ---")
r128_hits = []
for token in ("impulse", "controlled kernel", "Cumulative"):
    start = r126_notes.lower().find(token.lower())
    if start >= 0:
        r128_hits.append((start, token))
for start, token in sorted(r128_hits)[:3]:
    print(token, r126_notes[max(0, start-150):start+850])
```