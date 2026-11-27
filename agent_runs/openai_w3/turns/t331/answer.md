The carrier uses sharp gates and very large projections; their behavior outside recorded histories remains uncertain. Earlier output contraction improved doubled insulin while preserving factual fit. I’ll test a stronger contraction next. It is useful only if actual rollout errors shrink and factual NLL stays within the limit.

```python
alpha289 = 0.005
output_names289 = {
    "core.prelude.0.attn.out.weight",
    "core.prelude.0.ffn.2.weight",
    "core.prelude.0.ffn.2.bias",
}
for n in list(edited()):
    if n not in checkpoint273:
        drop_edit(n)
for n, x in checkpoint273.items():
    o = load_original(n)
    if hasattr(o, "detach"):
        o = o.detach().cpu().numpy()
    x = as_numpy(x)
    save_edit(n, o + alpha289 * (x-o) if n in output_names289 else x)
checkpoint289 = {n: as_numpy(load(n)).copy() for n in edited()}
print("Installed output-residual scale", alpha289, "relative to checkpoint273")

print("\nCOMPACT REPLAY HELPERS")
for name in ("original_tokens278", "parts254", "response_raw234"):
    fn = globals()[name]
    print(name, "arguments", fn.__code__.co_varnames[:fn.__code__.co_argcount])
    print("globals", fn.__code__.co_names)
    ops = [f"{i.opname}:{i.argrepr}" for i in dis.get_instructions(fn)]
    print(" | ".join(ops)[:1100])

print("\nCURRENT REPLAY ARRAY SHAPES")
for name in ("h155", "r233_parts", "r233_after", "r233_hidden", "r233_raw",
             "base_tokens278", "base_parts278", "dist255", "rates267", "dirs267"):
    v = globals().get(name)
    print(name, type(v).__name__,
          [(k, getattr(x, "shape", None)) for k, x in v.items()] if isinstance(v, dict)
          else getattr(v, "shape", None))

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 289; forwards 28. Installed reversible output-residual contraction alpha=.005 relative to checkpoint273; posterior, z_in and q/kv remain unchanged. Previous alpha=.02 passed factual (+.00175) and improved insulin x2 to -15.750 but left other readouts weak. Testing stronger contraction with factual safeguard, then paired rollout. checkpoint287 preserves gamma=10; checkpoint285 preserves alpha=.02.\n")
```