The edit uses a linear decoder of the normalized latent state, followed by a large transport direction. I’ll check the decoder’s held-out contrasts and the exact prelude calculation. If the decoder is roughly symmetric but the resulting state changes are not, I’ll revise the factual compensation rather than amplify the signal.

```python
import json

print("=== FORWARD SUMMARIES ===")
for k in [11, 12]:
    print(k, json.loads(Path(f"/work/fwd_{k}/summary.json").read_text()))

print("\n=== HELD-OUT LATENT CONTRAST ACCURACY ===")
valid196 = np.asarray(valid187, dtype=bool)
for d in [0, 1]:
    for s in [0.0, 0.5, 1.5, 2.0, 3.0]:
        dx, dy = pairs187[(d, s)]
        pred = np.asarray(dx) @ decoder192
        target = np.asarray(dy)
        use = valid196 & (np.abs(target[:, d]) > 1e-8)
        p, y = pred[use, d], target[use, d]
        slope = float(p @ y / (y @ y)) if len(y) else np.nan
        print(d, s, "n", len(y), "pred/target means",
              np.round([p.mean(), y.mean()], 5),
              "slope", round(slope, 4),
              "RMSE", round(float(np.sqrt(np.mean((p-y)**2))), 5),
              "cross-drug RMS", round(float(np.sqrt(np.mean(pred[use, 1-d]**2))), 5))
print("factual decoded RMS", np.sqrt(np.mean((xc187 @ decoder192)**2, axis=0)))

print("\n=== PRELUDE / CALIBRATION FUNCTION DETAILS ===")
for name in ["prelude155", "preludeE185", "hidden192", "response186"]:
    f = globals()[name]
    print(name, inspect.signature(f), "constants:",
          [x for x in f.__code__.co_consts if isinstance(x, (str, int, float, tuple))])
    print("dependencies:", f.__code__.co_names)

print("\n=== COMPACT PRELUDE WRAPPER BYTECODE ===")
for name in ["preludeE185", "hidden192"]:
    print(name)
    for ins in dis.Bytecode(globals()[name]):
        if ins.opname not in ["RESUME", "CACHE", "PRECALL", "PUSH_NULL"]:
            print(ins.opname, ins.argrepr)

print("\n=== TOKEN AND COMPENSATION INVENTORY ===")
for name in sorted(dict(globals())):
    if re.search(r"(tokens|after|zero169|154|169|compens)", name):
        v = globals()[name]
        if hasattr(v, "shape"):
            print(name, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(name, "dict", list(v)[:12])
        elif isinstance(v, (int, float, bool)):
            print(name, v)

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 196; forwards 12. Checking actual held-out paired latent contrasts and prelude compensation before tuning signal gains. Current pilot remains installed; no weights changed.\n")
```