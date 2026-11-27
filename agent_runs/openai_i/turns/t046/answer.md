The estimated effects persist for several hours, so a sensor-only output correction would likely miss the timing and secondary-sensor tests. I’ll investigate a small recurrent-state correction through the posterior. It is promising only if its features can distinguish a changed dose from the dose expected given the preceding condition.

```python
print("=== POSTERIOR AND TRANSITION TENSORS ===")
prefixes39 = (
    "core.z_", "core.gap", "core.flow", "core.out_norm",
    "core.prelude", "core.blocks.0.0", "core.coda",
    "core.delta", "core.a_in", "core.b_in", "core.c_out"
)
for name in NAMES:
    if name.startswith(prefixes39):
        print(name, SHAPES[name])

print("\n=== RECONSTRUCTION HELPERS ===")
for name in ["repair_atom_base", "repair_linear", "repair_norm",
             "repair_mlp_act", "head_exact"]:
    fn39 = globals()[name]
    print(name, inspect.signature(fn39))
    print("  variables:", fn39.__code__.co_varnames)
    print("  names:", fn39.__code__.co_names)
    print("  constants:", repr(fn39.__code__.co_consts)[:700])

print("\n=== POSTERIOR OBJECTS ===")
for name, obj in sorted(list(globals().items())):
    if not name.startswith("repair_") or callable(obj):
        continue
    if hasattr(obj, "shape"):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(name, "dict", len(obj), repr(list(obj.items())[:2])[:450])
    elif isinstance(obj, (list, tuple)):
        print(name, type(obj).__name__, len(obj), repr(obj[:1])[:220])
    elif isinstance(obj, (str, float, int)):
        print(name, repr(obj))

print("\n=== FORWARD FILES ===")
for folder39 in sorted(Path("/work").glob("fwd_*")):
    print(folder39.name, [p.name for p in folder39.iterdir()])

print("\n=== VOCAB TOP-LEVEL KEYS ===")
print(list(VOCAB))
print("config:", VOCAB.get("config"))
print("normalisation:", repr(VOCAB.get("normalisation", VOCAB.get("normalization")))[:1000])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 39; forward 6 — edit design\n"
            "- Lag estimates favor a recurrent correction rather than a sensor-only shortcut.\n"
            "- Inspecting exact posterior features and transition weights for a reversible low-rank innovation edit.\n"
            "- No weights changed.\n")
```