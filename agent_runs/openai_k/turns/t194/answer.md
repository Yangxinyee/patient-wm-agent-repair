This repeats run 177’s output, so there is no new feature-capture result to interpret. The reconstruction remains approximate and the saved weights are unchanged. I’ll run the compact capture check; usable paired hidden features would let us fit a correction, while missing or inconsistent features would rule out that route.

```python
import torch, numpy as np, re
from pathlib import Path

print("READOUT INTERFACE")
print("sensors", r49_sensors)
print("noise", r175_desc(r49_noise))
print("locals", r49_readouts.__code__.co_varnames)

r178_capture = []
r178_old_lin = lin0
def r178_capture_lin(x, prefix):
    if prefix in ("attr_heads.value_bin.0", "attr_heads.value_bin.2"):
        r178_capture.append((prefix, x.detach().clone()))
    return r178_old_lin(x, prefix)

r178_ix = np.arange(min(24, len(sx)))
try:
    lin0 = r178_capture_lin
    r178_result = r49_readouts(
        torch.tensor(sx[r178_ix, :192], dtype=torch.float32),
        torch.tensor(sx[r178_ix, 192:960].reshape(-1,4,192), dtype=torch.float32),
        r178_ix)
finally:
    lin0 = r178_old_lin

print("captured", [(p, tuple(x.shape)) for p,x in r178_capture])
print("result", r175_desc(r178_result))
if isinstance(r178_result, (tuple, list)):
    print("result parts", [r175_desc(x) for x in r178_result])

print("ACTION-CONTRAST OBJECTS")
r178_found = []
for name,obj in list(globals().items()):
    if re.search(r"(151|156|157|158|163|164|165|166|167)$", name) or re.match(r"r(151|156|157|158|163|164|165|166|167)_", name):
        if not name.startswith("__"):
            r178_found.append((name,obj))
for name,obj in r178_found[:55]:
    print(name, r175_desc(obj))

print("ACTION FUNCTIONS")
for name,obj in list(globals().items()):
    if callable(obj) and hasattr(obj,"__code__"):
        if any(s in name.lower() for s in ("action", "dose", "latent", "candidate")):
            print(name, obj.__code__.co_varnames[:obj.__code__.co_argcount],
                  "uses", obj.__code__.co_names[:12])

print("SCRATCH VERSUS ORIGINAL/SAVED")
for name in ("z_post.0.weight", "z_post.2.weight", "z_in.weight",
             "attr_heads.value_bin.0.weight", "attr_heads.value_bin.2.weight"):
    print(name, "scratch-original max",
          float((W0[name]-load_original(name)).abs().max()),
          "saved-original max", float((load(name)-load_original(name)).abs().max()))

with open("/work/notes.md","a") as f:
    f.write("\n## Run 178: paired-head feature capture\n"
            "- Repeated run-177 output supplied no new result; proceeding with capture.\n"
            "- Absolute local-head NLL remains non-authoritative; candidate acceptance "
            "will use official forward passes.\n"
            "- Capturing hidden head features and recovering physical action contrasts.\n"
            "- Python runs: 178; forward passes: 10. No saved changes.\n")
```
