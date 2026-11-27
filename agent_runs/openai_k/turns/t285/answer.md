The current candidate fails the factual-fit requirement: loss increased by 0.01016 nats, over twice the allowed increase. Most of the degradation is in MAP predictions, suggesting the norepinephrine component may be the main problem. I’ll separate the two corrections and recover their calibration code. If insulin alone preserves fit, we can retain it while rebuilding the norepinephrine correction.

```python
r254_original = load_original("core.b_in.weight").clone()
r254_current = load("core.b_in.weight").clone()
r254_delta = r254_current - r254_original
np.savez("/work/recurrent_half_backup.npz",
         original=r254_original.numpy(), edited=r254_current.numpy())

print("=== CURRENT CORRECTION ===")
print("norm", float(torch.linalg.vector_norm(r254_delta)),
      "singular values", torch.linalg.svdvals(r254_delta)[:6].tolist())

snap254 = list(globals().items())
print("\n=== RECENT PREFIXED OBJECTS ===")
for n, v in snap254:
    if not re.match(r"r(?:24[0-9]|250|251)_", n):
        continue
    if isinstance(v, (np.ndarray, torch.Tensor)):
        desc = f"{type(v).__name__}{tuple(v.shape)}"
        if v.numel() <= 8 if isinstance(v, torch.Tensor) else v.size <= 8:
            desc += " " + str(v.tolist())
    elif isinstance(v, dict):
        desc = "dict:" + ",".join(map(str, list(v)[:12]))
    elif inspect.isfunction(v):
        desc = "function:" + ",".join(inspect.signature(v).parameters)
    elif isinstance(v, (tuple, list)):
        desc = f"{type(v).__name__}[{len(v)}]"
    elif isinstance(v, (str, int, float, np.number)):
        desc = str(v)[:100]
    else:
        desc = type(v).__name__
    print(n, desc)

print("\n=== LAST 35 USER-DEFINED FUNCTIONS ===")
fun254 = [(n, v) for n, v in snap254
          if inspect.isfunction(v) and not n.startswith("_")]
for n, v in fun254[-35:]:
    print(n, list(inspect.signature(v).parameters))

print("\n=== SMALL CALIBRATION ARRAYS ===")
for n in ["slopes243", "slopes245", "r244_targets", "r251_strength"]:
    if n in globals():
        v = globals()[n]
        print(n, v.tolist() if isinstance(v, (np.ndarray, torch.Tensor)) else v)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 254: half-strength correction fails factual criterion\n"
        "- Forward 18 factual NLL original 1.63155439, edited 1.64171492, delta +0.01016053.\n"
        "- Event deltas: finger +0.00322, lab +0.00084, arterial +0.01930, cuff +0.01331; "
        "MAP dominates degradation.\n"
        "- Saved current candidate in recurrent_half_backup.npz; recovering separate drug "
        "components and replay interfaces before redesign. No weights changed.\n"
        "- Python runs: 254; forward passes: 19.\n"
    )
```
