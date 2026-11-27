The exact head-reconstruction helpers survived the restart, and the strength-2 insulin candidate remains intact. I can analyze the probe locally without another forward pass. I’ll verify that its states align with the cached readings, then measure changes in the context the heads actually use. A measurable, consistent direction would justify a constrained decoder; poor alignment or negligible changes would rule it out.

```python
print("--- Probe/cache alignment ---")
r127_aligned = all(
    np.array_equal(r125_original[k], r123_states[k])
    for k in ("stay_id", "hour", "has_insulin", "has_norepinephrine", "n_atoms")
)
print("Same bundles as cached reconstruction:", r127_aligned)
if r127_aligned:
    print("Original-state max difference:",
          float(np.max(np.abs(r125_original["x"] - r123_states["x"]))))
    r127_ctx_o = r124_context(r125_original, meta96)
    r127_ctx_e = r124_context(r125_probe, meta96)
    r127_dctx = (r127_ctx_e - r127_ctx_o).numpy()
    print("Context reconstruction max error:",
          float(torch.max(torch.abs(r127_ctx_o - r124_ctx))))
    for ev in sorted(meta96.event.unique()):
        m = meta96.event.to_numpy() == ev
        print(ev, "n", int(m.sum()),
              "context delta norm mean/max",
              float(np.linalg.norm(r127_dctx[m], axis=1).mean()),
              float(np.linalg.norm(r127_dctx[m], axis=1).max()))

print("\n--- Critical helper definitions ---")
for name in ("r124_context", "r124_lin", "r124_mlp", "r124_norm", "local104"):
    func = globals()[name]
    try:
        text = inspect.getsource(func)
        print(name, "\n", text[:1700])
    except (OSError, TypeError):
        print(name, "referenced names:", func.__code__.co_names,
              "constants:", [x for x in func.__code__.co_consts
                             if isinstance(x, (str, int, float))][:25])

print("\n--- Candidate/reconstruction variable names ---")
for prefix in ("r98", "r99", "r103", "r104", "r105", "r106"):
    print(prefix, sorted(k for k in globals() if k.startswith(prefix)))
print("Other reconstruction arrays:",
      [(k, tuple(v.shape)) for k, v in list(globals().items())
       if hasattr(v, "shape") and
       any(t in k.lower() for t in ("head_input", "draw", "gate", "latent"))])

print("\n--- Initial record-derived evidence ---")
print(r126_notes[:3600])
```