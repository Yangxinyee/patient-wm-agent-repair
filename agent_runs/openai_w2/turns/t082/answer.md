The recovered notes rule out a head-only correction: it could not recover enough dose response while staying quiet on factual records. I’ll focus on injecting a dose residual upstream, where it can affect subsequent readings. This run only inventories the helpers and saved states needed to test that idea.

```python
print("=== USER-DEFINED FUNCTIONS ===")
print([
    k for k, v in list(globals().items())
    if not k.startswith("_") and callable(v)
    and getattr(v, "__module__", "") == "__main__"
])

print("\n=== RELEVANT VARIABLES ===")
for k, v in list(globals().items()):
    if k.startswith("_"):
        continue
    if any(s in k.lower() for s in (
        "state", "reference", "panel", "train", "tuning", "action",
        "posterior", "bundle", "atom", "head_cfg", "repair_", "benchmark"
    )):
        if isinstance(v, (np.ndarray, torch.Tensor, pd.DataFrame)):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "dict", list(v)[:10])
        elif isinstance(v, (list, tuple)):
            print(k, type(v).__name__, len(v))
        elif not callable(v):
            print(k, str(v)[:160])

print("\n=== PATHWAY TENSORS ===")
for n in NAMES:
    if any(s in n for s in ("z_post", "z_in", "encode_action", "core.b_in")):
        print(n, SHAPES[n])

print("\n=== SAVED STATES ===")
for p in sorted(Path("/work").glob("fwd_*/states*.npz")):
    with np.load(p) as f:
        print(p, {k: f[k].shape for k in f.files})

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Runs 66–68: third context recovery\n"
        "No edits. Observation/END posterior reconstruction is still inaccurate, "
        "but prior diagnostics reconstructed single-drug bundles to about 3e-5. "
        "Head-only factual-zero dose decoders were previously rejected. "
        "Investigating a residual-dose injection into the latent/recurrent pathway, "
        "with held-out factual checks before accepting any edit.\n"
    )
```
