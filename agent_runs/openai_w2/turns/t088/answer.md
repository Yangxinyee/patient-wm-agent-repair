The architecture confirms that dose changes must pass through the action encoder and recurrent state. Earlier amplification edits did not reliably strengthen that pathway. I’ll inspect the reconstructed action features next; a selective correction is promising only if it distinguishes changed doses while remaining small on held-out factual bundles.

```python
import inspect
import re

print("=== RELEVANT WEIGHT SHAPES ===")
for name in NAMES:
    if any(s in name for s in (
        "z_post", "z_in", "encode_action", "core.prelude", "attr_heads.value_bin"
    )):
        print(name, SHAPES[name])

print("\n=== RECONSTRUCTION VARIABLES ===")
for name, value in sorted(dict(globals()).items()):
    if name.startswith("_") or not re.search(
        r"repair|posterior|post_hidden|hidden_vari|state_meta|bundle_meta|^states|^state_", name
    ):
        continue
    if isinstance(value, pd.DataFrame):
        print(name, value.shape, list(value.columns))
    elif isinstance(value, (np.ndarray, torch.Tensor)):
        print(name, type(value).__name__, tuple(value.shape))
    elif isinstance(value, dict):
        print(name, "dict:", [
            (str(k), tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
            for k, v in list(value.items())[:12]
        ])
    elif callable(value):
        try:
            print(name, inspect.signature(value),
                  "uses:", value.__code__.co_names)
        except Exception:
            print(name, type(value).__name__)
    elif isinstance(value, (str, int, float, list, tuple)):
        print(name, str(value)[:180])

print("\n=== CORE HELPER DEFINITIONS ===")
for name in ("previous_context_repair", "action_encode_variant",
             "embed_record_atom", "encode_variable_sets_repair",
             "local_value_head"):
    value = globals().get(name)
    if value is not None:
        try:
            print(name, inspect.getsource(value))
        except Exception:
            print(name, "source unavailable; constants:",
                  repr(value.__code__.co_consts)[:1600])

print("\n=== EARLY NOTEBOOK SUMMARY ===")
notes = Path("/work/notes.md").read_text()
sections = re.split(r"(?m)(?=^## )", notes)
for section in sections:
    if any(s in section.lower() for s in (
        "head-only", "probe", "regression discontinuity", "threshold", "causal"
    )):
        print(section[:2200])
        if sum(len(s[:2200]) for s in sections[:sections.index(section)+1]
               if any(x in s.lower() for x in
                      ("head-only", "probe", "regression discontinuity", "threshold", "causal"))) > 4800:
            break
```
