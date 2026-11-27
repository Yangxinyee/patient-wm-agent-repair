The inspection was cut off by a verbose bytecode dump, so it supplied no new validation evidence. I’ll switch to bounded output and retrieve only the configuration, recurrent tensors, and cached head arrays. Those should determine whether a local replay is feasible without another broad reconstruction search.

```python
print("=== CONFIGURATION ===")
print(rr_vocab["config"])
print("\n=== RECORD SCHEMA ===")
print(rr_train.head(6).to_string(index=False))

print("\n=== RECURRENT TENSOR LAYOUT ===")
for n in NAMES:
    if n.startswith("core."):
        print(n, SHAPES[n])

print("\n=== AVAILABLE EXPORTS ===")
for p in sorted(Path("/work").glob("fwd_*/*")):
    print(str(p))

print("\n=== CACHED HEAD OBJECTS ===")
for k in sorted(list(globals())):
    if k.startswith("ce_"):
        v = globals()[k]
        if isinstance(v, (np.ndarray, torch.Tensor, pd.DataFrame)):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "dict", [(a, tuple(b.shape) if hasattr(b, "shape") else type(b).__name__) for a, b in list(v.items())[:12]])
        elif isinstance(v, (list, tuple)):
            print(k, type(v).__name__, len(v))

print("\n=== SMALL HELPER CONSTANTS ===")
for k in ["rr_linear", "rr_norm", "rr_mlp", "rr_encode_atoms112", "rr_atom"]:
    f = globals()[k]
    print(k, "names=", f.__code__.co_names)
    print("constants=", [
        c for c in f.__code__.co_consts
        if isinstance(c, (str, int, float, tuple)) and len(str(c)) < 180
    ])

print("\n=== HEAD COMPUTATION, COMPACT BYTECODE ===")
rx_instr = list(dis.get_instructions(ce_make_heads))
rx_start = next((i for i, x in enumerate(rx_instr) if x.argval == "core.flow.a"), len(rx_instr))
for ins in rx_instr[max(0, rx_start-8):rx_start+145]:
    if ins.opname not in {"CACHE", "RESUME", "EXTENDED_ARG"}:
        print(ins.offset, ins.opname, str(ins.argrepr)[:100])

with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 123\nRun 122 output was truncated by disassembly; no new model validation or edit resulted. Using bounded diagnostics for configuration, recurrent weights, and cached head arrays. No edits. Counts: 123 Python runs, 6 forward passes.\n")
```