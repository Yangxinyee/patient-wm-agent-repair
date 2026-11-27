The visible embedding helper follows the documented binning and normalization rules. I’ll check context timing and the observation encoder next, since either could produce the posterior errors. Matching these calculations against saved states will distinguish an encoding problem from a state-alignment problem.

```python
print("=== CONTEXT AND ENCODER LOGIC ===")
for name in ["previous_context_repair", "rmlp", "rnorm", "batched_set_repair"]:
    fn = globals().get(name)
    if fn is None:
        continue
    instructions = list(dis.get_instructions(fn))
    print("\n", name, "constants:", fn.__code__.co_consts)
    print(" | ".join(f"{i.opname} {i.argrepr}" for i in instructions))

print("\n=== RELEVANT WEIGHTS ===")
for name in NAMES:
    if name.startswith(("embed.categorical.", "encode.", "z_post.", "core.gap")):
        print(name, SHAPES[name])

print("\n=== RECONSTRUCTION DATA ===")
for name, obj in sorted(list(globals().items())):
    if name.startswith("_") or not any(
        s in name.lower() for s in ["repair", "recon", "bundle", "state", "post"]
    ):
        continue
    if isinstance(obj, pd.DataFrame):
        print(name, obj.shape, list(obj.columns))
    elif isinstance(obj, dict):
        print(name, "dict", list(obj)[:10])
    elif hasattr(obj, "shape"):
        print(name, tuple(obj.shape))
    elif hasattr(obj, "files"):
        print(name, "npz", obj.files)
print("Current edits:", edited())
```