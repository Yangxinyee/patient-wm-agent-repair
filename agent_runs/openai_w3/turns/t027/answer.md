The live kernel still contains the earlier reconstruction helpers and causal estimates. I’ll check the atom-building code directly, especially bin boundaries and default embeddings. If a small encoding correction makes measurement posteriors match, we can proceed to editing; otherwise, we should not trust the local reconstruction.

```python
import inspect, dis

print("=== HELPER IMPLEMENTATIONS ===")
for fname, limit in [
    ("make_atom", 4200),
    ("atom_base", 2200),
    ("encode_one_set", 1200),
    ("encode_single_atom", 1000),
]:
    func = globals()[fname]
    try:
        text = inspect.getsource(func)
    except Exception:
        text = dis.Bytecode(func).dis()
    print("\n" + fname + ":\n" + text[:limit])

print("\n=== WEIGHT TYPES ===")
for name in ("embed.numeric.weight", "embed.kq.weight", "z_post.0.weight"):
    if name in W:
        print(name, type(W[name]), tuple(W[name].shape))

print("\n=== REMAINING EMBEDDING / LATENT TENSORS ===")
for name in NAMES:
    if name.startswith(("embed.categorical.", "embed.kq", "embed.text", "z_post", "z_prior", "z_head", "gap", "encode_action", "encode.")):
        print(name, SHAPES[name])

print("\n=== SMALL DIAGNOSTIC OBJECTS ===")
print("history available:", "In" in globals(), "get_ipython" in globals())
print("isolated map_cuff:", repr(isolated_by_event["map_cuff"])[:500])
print("default vector norm:", float(np.linalg.norm(as_numpy(defaults))))
print("edited:", edited())
```