The old calibration matrix is mostly at float32 rounding scale, so its inverse cannot justify an edit. I’ll first check whether the local readout responds meaningfully to state perturbations. A stable, nonzero response would support recalibration; a constant response would expose a replay bug.

```python
import numpy as np, torch, dis

print("READOUT IMPLEMENTATION")
compact183("readout134", 3800)
print("\nREADOUT GLOBALS")
for name in readout134.__code__.co_names:
    if name not in globals():
        continue
    x = globals()[name]
    if hasattr(x, "shape"):
        print(name, tuple(x.shape))
    elif isinstance(x, dict):
        print(name, "dict", list(x)[:8])
    elif callable(x):
        print(name, "function")
    else:
        print(name, str(x)[:100])

print("\nREPLAY INPUTS")
for name in ["hf169", "hh169", "mem169", "mask169", "age169", "age_emb169",
             "dt169", "i169", "C154", "E154", "U154", "Z154",
             "raw_latent154", "normalized_latent154", "post177",
             "raw177", "ns177", "zs177", "features177"]:
    if name in globals():
        x = globals()[name]
        print(name, type(x).__name__, tuple(x.shape) if hasattr(x, "shape") else str(x)[:100])

print("\nACTION HELPER RETURN")
for ins in list(dis.get_instructions(action_vectors87))[-20:]:
    print(ins.opname, ins.argrepr)

# Test the local head independently of the recurrent replay.
probe184 = tensor153(np.random.default_rng(184).normal(size=(16, 192)))
test184 = {}
for scale in [0.0, 0.1, 1.0, 5.0]:
    val = readout134((probe184 * scale)[:, None, :])
    test184[scale] = arr183(val)
    print("readout probe", scale, test184[scale].shape,
          "mean/std", float(test184[scale].mean()), float(test184[scale].std()))
for scale in [0.1, 1.0, 5.0]:
    print("paired change", scale,
          float(np.sqrt(np.mean((test184[scale] - test184[0.0]) ** 2))))

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 184; forwards 10. Old J171 is dominated by float32 quantization and is not a trusted physiology calibration. Testing readout sensitivity independently before any further weight edit. Five zero-payload transport edits remain unchanged.\n")
```
