The architecture confirms that dose information reaches the shared recurrent state before the value head. There is no separate drug-effect parameter to adjust. I’ll recover the record-based effect estimates and inspect the value-head helpers. A direct head repair remains plausible only if dose changes produce detectable state contrasts that can be separated from factual prediction features.

```python
import inspect
from pathlib import Path

print("=== EARLY CAUSAL EVIDENCE ===")
print(Path("/work/notes.md").read_text()[:5500])

print("\n=== RELEVANT WEIGHT SHAPES ===")
for name in NAMES:
    if any(s in name for s in (
        "attr_heads.value", "last_numeric", "query_norm", "type_emb",
        "core.c_out", "core.out_norm", "core.z_in", "core.z_head",
        "embed_transition.value"
    )):
        print(name, SHAPES[name])

print("\n=== HELPER CONTRACTS ===")
for name in ["readout153", "readout_capture156", "capture159",
             "replay_inputs161", "call_core161", "core_step136",
             "encode_actions168", "features169", "asnp175"]:
    obj = globals().get(name)
    if callable(obj) and hasattr(obj, "__code__"):
        print(name, "locals:", obj.__code__.co_varnames)
        print("constants:", repr(obj.__code__.co_consts)[:650])

print("\n=== CANDIDATE / CAPTURE VARIABLES ===")
for name, obj in list(globals().items()):
    if name.startswith("_"):
        continue
    if any(s in name.lower() for s in
           ["capt", "replay", "lag", "candidate", "head", "factual", "panel", "record", "direction"]):
        if isinstance(obj, dict):
            print(name, "dict keys:", list(obj.keys())[:12])
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            print(name, tuple(obj.shape))
        elif hasattr(obj, "columns"):
            print(name, obj.shape, list(obj.columns))
        elif isinstance(obj, str):
            print(name, "string length", len(obj))
```
