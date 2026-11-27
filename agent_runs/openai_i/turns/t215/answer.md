Simply enlarging action vectors was already tested and did not strengthen the effects. I’ll now measure the action information that reaches the recurrent update itself. If a dose-innovation signal can be decoded there on held-out stays, a low-rank recurrent edit may work; poor decoding would argue against this pathway.

```python
import sys, dis
torch.set_num_threads(8)

print("=== CORE IMPLEMENTATION CONTRACT ===")
print("locals:", core_step136.__code__.co_varnames)
print("constants:", core_step136.__code__.co_consts)
for n191 in ["linear136", "norm136", "mlp136", "block136", "attn136", "batch_replay182"]:
    o191 = globals().get(n191)
    if callable(o191):
        print(n191, inspect.signature(o191), "refs", o191.__code__.co_names,
              "locals", o191.__code__.co_varnames)

captures191 = []
def profile191(frame, event, arg):
    if event == "return" and frame.f_code is core_step136.__code__:
        loc = frame.f_locals
        if not captures191:
            print("\n=== FIRST CORE CALL ===")
            for k, v in loc.items():
                if torch.is_tensor(v):
                    print(k, tuple(v.shape), "RMS", float(v.square().mean().sqrt()))
                elif isinstance(v, (str, int, float)):
                    print(k, v)
            print("caller locals:",
                  {k: (tuple(v.shape) if hasattr(v, "shape") else
                       len(v) if isinstance(v, (list, tuple, dict)) else str(v)[:80])
                   for k, v in frame.f_back.f_locals.items()})
        captures191.append({
            k: v.detach().clone()
            for k, v in loc.items()
            if k in ["E", "h_before", "h", "e", "u", "z", "dt_seconds"] and torch.is_tensor(v)
        })
previous_profile191 = sys.getprofile()
try:
    sys.setprofile(profile191)
    base_h191, base_o191 = batch_replay182(Uref168)
finally:
    sys.setprofile(previous_profile191)
print("\nCaptured calls:", len(captures191))
print("First captured keys:", list(captures191[0]))
print("Replay shapes:", base_h191.shape, base_o191.shape)
print("args_cache182 type:", type(args_cache182))
if isinstance(args_cache182, dict):
    k191 = next(iter(args_cache182))
    v191 = args_cache182[k191]
    print("first cache key:", k191, "value type:", type(v191))
    if isinstance(v191, (tuple, list)):
        print("entry shapes/types:",
              [(tuple(x.shape) if hasattr(x, "shape") else type(x).__name__) for x in v191])

with open("/work/notes.md", "a") as f191:
    f191.write("\n## Python 191 — expose recurrent-input features\n"
               "- Capturing the prelude E that feeds core.b_in, without changing weights or using gradients.\n"
               "- Will test dose-innovation decoding and downstream response before any installation.\n")
```