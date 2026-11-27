The records support a negative insulin effect, but the early regressions are not yet a multi-hour rollout target. I’ll recover the later dynamic estimates and the relay layout. The next candidate will retain the factual-safe latent mapping and change only how its signal reaches the state.

```python
import numpy as np
from pathlib import Path

def r243_meta(x, depth=0):
    if isinstance(x, np.ndarray):
        return {"shape": x.shape, "dtype": str(x.dtype)}
    if isinstance(x, dict):
        return {str(k): r243_meta(v, depth+1) for k, v in list(x.items())[:18]} if depth < 2 else list(x)[:18]
    if isinstance(x, (list, tuple)):
        return [r243_meta(v, depth+1) for v in x[:6]] if depth < 2 else f"{type(x).__name__}[{len(x)}]"
    return str(x)[:220]

print("DYNAMIC NOTES")
r243_notes = Path("/work/notes.md").read_text()
print(r243_notes[7400:11300])

print("\nESTIMATES")
if "causal_estimates" in globals():
    print(str(causal_estimates)[:1500])

print("\nRELAY AND SELECTOR DEFINITIONS")
for r243_name in ("relay206", "relay207_base", "selector240", "cf239", "r228", "cache", "inj_cache"):
    if r243_name in globals():
        print(r243_name, r243_meta(globals()[r243_name]))

print("\nAVAILABLE CACHE AND COMPUTATION NAMES")
print(", ".join(
    n for n, x in list(globals().items())
    if not n.startswith("r24")
    and (any(s in n.lower() for s in ("cache", "metadata", "forward", "local_step", "view221"))
         or (callable(x) and not n.startswith("_") and hasattr(x, "__code__")))
)[-2600:])

print("\nCURRENT ATTENTION LAYOUT")
for r243_name in ("core.prelude.0.attn.q.weight", "core.prelude.0.attn.kv.weight",
                  "core.prelude.0.attn.out.weight"):
    r243_w = np.asarray(load(r243_name))
    r243_o = np.asarray(load_original(r243_name))
    print(r243_name, r243_w.shape,
          "diff_norm", float(np.linalg.norm(r243_w-r243_o)))
    if r243_name.endswith("out.weight"):
        print("column-block norms", np.linalg.norm(r243_w.reshape(192,4,48), axis=(0,2)).round(3))
    elif r243_w.shape[0] == 192:
        print("row-block norms", np.linalg.norm(r243_w.reshape(4,48,192), axis=(1,2)).round(3))

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 243\nRetrieved later dynamic-identification evidence and compensated-relay layout. Safe236 remains unchanged; selective relay will retain its z_in mapping. Counts: 243 Python runs, 23 forwards.\n")
```