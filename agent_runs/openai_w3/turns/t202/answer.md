The records support substantial insulin and norepinephrine effects, while the current model’s rollouts are much weaker. The reserved attention head is factual-neutral, but its dose payload remains unvalidated. I’ll inspect the posterior and transport equations next; the key test is whether they preserve paired dose changes without disturbing factual states.

```python
import inspect, dis, io, numpy as np

def show182(name, limit=2400):
    obj = globals().get(name)
    if obj is None:
        return
    print("\n---", name, "---")
    try:
        src = inspect.getsource(obj)
    except Exception:
        stream = io.StringIO()
        dis.dis(obj, file=stream, depth=0)
        src = stream.getvalue()
    print(src[:limit])

for name in ["posterior153", "prelude155", "action_vectors87",
             "state_update82", "view171", "phys171"]:
    show182(name)

print("\nTRANSPORT AND CANDIDATE DETAILS")
for name in ["reserved154", "value_extract154", "carrier154", "carrier_scale154",
             "payload154", "policy154", "presence154", "rate154", "rmse154",
             "signal169", "payload_kv171", "value_carrier171", "head174",
             "gatevec177", "rates177", "features177", "scenario177"]:
    if name not in globals():
        continue
    x = globals()[name]
    if isinstance(x, dict):
        print(name, [(k, np.shape(v)) for k, v in x.items()])
    elif hasattr(x, "shape"):
        a = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        print(name, a.shape, "range", (float(np.min(a)), float(np.max(a))),
              "norm", float(np.linalg.norm(a)), "sample", a.ravel()[:8])
    else:
        print(name, str(x)[:600])

print("effect171", effect171)
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 182; forwards 10. Recovered early record-derived impulse kernels and tuning references; substantial effects are evidence-based, not clinical assumptions. Inspecting posterior and transport equations to isolate contrast attenuation. Existing zero-payload edits unchanged.\n")
```
