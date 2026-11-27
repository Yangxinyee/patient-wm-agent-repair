The core helpers and aligned state arrays are available, but the output was cut off by a binary-key lookup table. No new encoder test ran. I’ll now inspect only the encoding function’s dependencies and atom metadata, with strict output limits. This should provide everything needed for the unit-ID test without another broad inventory.

```python
import inspect

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 85\n"
        "Run 84 recovered core helpers but output was truncated by binary lookup "
        "keys. Restricting inspection to encoder dependencies and atom metadata. "
        "No weights changed. Counts: 85 Python runs, 6 forward passes.\n"
    )

print("=== EARLY EXPERIMENT SUMMARY ===")
print(Path("/work/notes.md").read_text()[:3900])

def r85_brief(x):
    if hasattr(x, "shape"):
        return f"{type(x).__name__} {tuple(x.shape)}"
    if callable(x):
        try:
            return str(inspect.signature(x))
        except Exception:
            return "callable"
    if isinstance(x, dict):
        return f"dict({len(x)})"
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__}({len(x)}) sample={repr(x[:2])[:220]}"
    return repr(x)[:120]

print("\n=== ENCODER DEPENDENCIES ===")
print("r79_encode names:", r79_encode.__code__.co_names)
print("r79_encode constants:", repr(r79_encode.__code__.co_consts)[:900])
for r85_name in r79_encode.__code__.co_names:
    if r85_name in globals() and r85_name.startswith(("r", "rr")):
        print(r85_name, r85_brief(globals()[r85_name]))

print("\n=== ATOM METADATA ===")
for r85_name in sorted(list(globals())):
    if r85_name.startswith("r79_"):
        print(r85_name, r85_brief(globals()[r85_name]))
for r85_key, r85_val in r79_masks.items():
    print("mask", r85_key, r85_brief(r85_val))
    if isinstance(r85_val, (list, tuple)):
        print(" components:", [r85_brief(x) for x in r85_val])

print("\n=== ATOM EMBEDDING HELPERS ===")
r85_helpers = []
for r85_name, r85_obj in list(globals().items()):
    if hasattr(r85_obj, "__code__"):
        r85_const = repr(r85_obj.__code__.co_consts)
        if "embed.categorical.unit.weight" in r85_const:
            r85_helpers.append(r85_name)
            print(r85_name, r85_brief(r85_obj))
            print(" names:", r85_obj.__code__.co_names)
            print(" locals:", r85_obj.__code__.co_varnames)
            print(" constants:", r85_const[:1600])
print("matching helpers:", r85_helpers)
```