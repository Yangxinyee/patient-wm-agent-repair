The last executed run verified the function interfaces and confirmed that the current edit strengthens insulin responses, but it did not explain their small size. I will recover the exact transition inputs and record-based targets before tracing an isolated dose impulse. A correct initial response that quickly disappears would implicate recurrent dynamics; a weak initial response would implicate the input channel.

```python
import inspect, numpy as np, torch
from pathlib import Path

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

snapshot151 = dict(globals())
def compact151(v, depth=0):
    if isinstance(v, (np.ndarray, torch.Tensor)):
        a = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
        if a.size <= 26:
            return {"shape": a.shape, "values": np.round(a, 5).tolist()}
        return {"shape": a.shape, "dtype": str(a.dtype)}
    if isinstance(v, dict):
        if depth >= 2:
            return {"keys": list(v)[:30]}
        return {str(k): compact151(x, depth+1) for k, x in list(v.items())[:30]}
    if isinstance(v, (list, tuple)):
        return [compact151(x, depth+1) for x in v[:16]]
    if hasattr(v, "shape"):
        return {"type": type(v).__name__, "shape": v.shape}
    return repr(v)[:160]

print("\n=== REPLAY BANK AND CHANNEL ===")
for name in ["repair_bank", "chosen_channel116", "lp_new136", "lp_new137",
             "scale_results128", "nll_scales129"]:
    if name in snapshot151:
        print(name, compact151(snapshot151[name]))

print("\n=== RECENT DIAGNOSTIC OBJECTS ===")
for name, obj in snapshot151.items():
    if any(name.endswith(str(i)) for i in [137, 147, 148, 149, 150]):
        if not inspect.isfunction(obj):
            print(name, compact151(obj))

print("\n=== EXACT CALLABLE INTERFACES ===")
functions151 = {}
for name, obj in snapshot151.items():
    if inspect.isfunction(obj) and (
        name in ["transition73", "transition_with_b110", "replay_factual125",
                 "probe_means95", "head_features_fixed"]
        or any(s in name.lower() for s in ["pre_context", "action", "encode", "head_features"])
    ):
        functions151[name] = obj
        print(name, inspect.signature(obj), "dependencies:", obj.__code__.co_names)
        if name in ["transition73", "transition_with_b110", "replay_factual125"]:
            try:
                print(inspect.getsource(obj))
            except (OSError, TypeError):
                import dis
                dis.dis(obj)

print("\nSaved edits:", edited())
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 151\n"
        "Recovered exact transition/replay interfaces, architecture, retained channel "
        "and causal-kernel objects for a bounded impulse diagnostic. Run 150 verified "
        "return structures only; it did not establish input compression or causal truth. "
        "No saved weights changed.\n"
    )
```