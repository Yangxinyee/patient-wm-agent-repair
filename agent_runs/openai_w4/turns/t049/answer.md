The architecture confirms that drug effects must pass through the recurrent state, while a separate last-reading shortcut helps factual prediction. Accurate posterior reconstruction therefore validates only the input stage.

I’ll inspect the retained action code and causal diagnostics without repeating the architecture. A useful repair must strengthen dose sensitivity while preserving observed-record behavior; otherwise it risks improving rollouts at the expense of factual fit.

```python
import inspect, re
from pathlib import Path

print("=== INITIAL IDENTIFICATION NOTES ===")
r41_notes = Path("/work/notes.md").read_text()
print(r41_notes[:3600])

print("\n=== RECENT RETAINED OBJECTS ===")
r41_items = [
    (n, v) for n, v in list(globals().items())
    if not n.startswith("_")
    and not n.startswith(("r40_", "r41_"))
    and not inspect.ismodule(v)
    and not callable(v)
]
for r41_name, r41_obj in r41_items[-85:]:
    if hasattr(r41_obj, "shape"):
        print(r41_name, type(r41_obj).__name__, tuple(r41_obj.shape))
    elif isinstance(r41_obj, dict):
        print(r41_name, "dict", repr(list(r41_obj.keys())[:8])[:130])
    elif isinstance(r41_obj, (list, tuple)):
        print(r41_name, type(r41_obj).__name__, len(r41_obj),
              repr(r41_obj[:2])[:130])
    else:
        print(r41_name, repr(r41_obj)[:130])

print("\n=== RECONSTRUCTION FUNCTION SOURCES ===")
r41_remaining = 2800
for r41_name in ["rp_linear", "rp_norm", "rp_mlp", "rp_action_encoder"]:
    r41_obj = globals().get(r41_name)
    if r41_obj is None:
        continue
    try:
        r41_source = inspect.getsource(r41_obj)
    except Exception as r41_exc:
        r41_source = (
            f"Source unavailable: {r41_exc}; "
            f"names={r41_obj.__code__.co_names}; "
            f"constants={r41_obj.__code__.co_consts}"
        )
    r41_piece = f"\n{r41_name}:\n{r41_source}"
    print(r41_piece[:r41_remaining])
    r41_remaining -= min(len(r41_piece), r41_remaining)
    if r41_remaining <= 0:
        break

print("\n=== TRANSITION TENSOR SHAPES ===")
for r41_name in NAMES:
    if any(s in r41_name for s in [
        "core.prelude", "core.coda", "core.flow",
        "core.b_in", "core.c_out", "core.delta_in", "core.a_in"
    ]):
        print(r41_name, SHAPES[r41_name])

with open("/work/notes.md", "a") as r41_file:
    r41_file.write(
        "\n## Run 41\n"
        "Architecture separates the explicit action encoder, recurrent transition, "
        "and same-code last-reading shortcut. Recovering recent reconstruction "
        "objects and initial causal diagnostics with compact output. No edits.\n"
    )
```
