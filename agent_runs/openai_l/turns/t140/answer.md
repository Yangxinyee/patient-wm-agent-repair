The recurrence is now clear. A simpler first candidate could add the missing immediate drug response and let the model’s existing reading feedback carry it forward, avoiding a new memory controller. I’ll inspect the saved private-channel and value-head sensitivities before testing that approach. If its delayed response fails to match the record kernels, explicit memory will be necessary.

```python
import re, dis, numpy as np, torch

parts131 = re.split(r"(?m)^## (.+)\n", open("/work/notes.md").read())
print("=== PRIOR HEAD AND CHANNEL EXPERIMENTS ===")
budget131 = 3300
wanted131 = {81, 83, 87, 89, 95, 96, 97, 98, 100, 110, 111, 112, 118}
for heading, body in zip(parts131[1::2], parts131[2::2]):
    m = re.search(r"run\s+(\d+)", heading, re.I)
    if m and int(m.group(1)) in wanted131:
        text = "\n## " + heading + "\n" + body.strip()[:380]
        print(text[:budget131])
        budget131 -= min(len(text), budget131)
        if budget131 <= 0:
            break

print("\n=== ACTIVE CPU HELPERS ===")
for name in ("linear63", "norm63", "attention63", "decode67",
             "linear_before125", "capture_pre125"):
    f = globals()[name]
    print(name, "actual:", f.__name__, "globals:", f.__code__.co_names)
    print("constants:", repr(f.__code__.co_consts)[:500])

print("\n=== PRIVATE STATE AND HEAD DIRECTIONS ===")
def array131(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

for name in ("Vsignal118", "R118", "Jselected118", "Jhead118",
             "T125", "QE125", "SP125", "Jpre125"):
    a = array131(globals()[name])
    print(name, a.shape,
          "column norms", np.round(np.linalg.norm(a, axis=0), 4)
          if a.ndim == 2 and a.shape[1] <= 4 else
          "RMS=" + str(round(float(np.sqrt(np.mean(a*a))), 6)))

print("Jselected118 @ R118:", np.round(array131(Jselected118) @ array131(R118), 4))
print("Vsignal118.T @ R118:", np.round(array131(Vsignal118).T @ array131(R118), 4))
print("T125 minus mean(Jpre125):",
      np.linalg.norm(array131(T125) - array131(Jpre125).mean(axis=0)))
print("private state factual RMS:",
      np.sqrt(np.mean((array131(Hpost118).reshape(-1,192)
                       @ array131(Vsignal118))**2, axis=0)))
print("head input factual RMS:",
      np.sqrt(np.mean((array131(xf118) @ array131(R118))**2, axis=0)))

print("\n=== RELEVANT WEIGHT SHAPES ===")
for name in NAMES:
    if any(s in name for s in ("attr_heads.value_bin", "core.b_in", "core.c_out",
                              "core.prelude.0.attn", "core.delta_in", "core.a_in")):
        print(name, SHAPES[name])
print("\n=== GENERIC INPUT / DECODER INTERFACES ===")
for name in ("bundle_inputs74", "decode67", "head_inputs96"):
    f = globals()[name]
    print(name, "constants:", repr(f.__code__.co_consts)[:1100])
    print("globals:", f.__code__.co_names)
```
