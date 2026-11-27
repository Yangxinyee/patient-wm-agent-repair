The earlier carrier-plus-relay passed this factual test: NLL improved slightly, by 0.00022 nats. That localizes the failure to the dedicated marker construction. I’ll retain the safe latent mapping and look for dose-sensitive projections of its existing tokens. The crucial check is whether those projections distinguish dose edits while having very little factual variation.

```python
print("Z_IN COLUMN DIFFERENCES: rejected marker versus factual-safe carrier")
d235 = arr231(rejected230["z_in.weight"]) - arr231(candidate211_weights["z_in.weight"])
for j in np.argsort(np.linalg.norm(d235, axis=0))[::-1][:8]:
    print(int(j), "difference", float(np.linalg.norm(d235[:, j])),
          "safe norm", float(np.linalg.norm(arr231(candidate211_weights["z_in.weight"])[:, j])),
          "marker norm", float(np.linalg.norm(arr231(rejected230["z_in.weight"])[:, j])))

print("\nPOSTERIOR HELPER PROBE")
probe235 = post208(np.zeros((1, 576), dtype=np.float32), candidate211_weights)
print(type(probe235).__name__)
if isinstance(probe235, tuple):
    print([brief230(x) for x in probe235])
else:
    print(brief230(probe235))

print("\nCOUNTERFACTUAL CACHES")
for k in sorted(list(globals())):
    if not k.startswith("_") and (
        k.startswith("rx_") or k.startswith("cf") or
        any(k.endswith(str(r)) for r in [178,180,200,205,208,211,212,213,215,228,229])):
        v = globals()[k]
        if isinstance(v, dict):
            print(k, brief230(v))
        elif torch.is_tensor(v) or isinstance(v, np.ndarray):
            print(k, tuple(v.shape))
        elif callable(v) and hasattr(v, "__code__"):
            print(k, "args", v.__code__.co_varnames[:v.__code__.co_argcount])
print("\nSAFE TOKEN/ENCODER HELPERS")
for k in sorted(list(globals())):
    v = globals()[k]
    if callable(v) and hasattr(v, "__code__") and any(
        w in k.lower() for w in ["counter", "scaled", "encode", "atoms", "embedding", "cf_", "tokens"]):
        print(k, v.__code__.co_varnames[:v.__code__.co_argcount])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 235\nForward23 carrier211+relay factual delta -.00022 on 2879 readings. "
            "Failure localized to dedicated marker construction. Plan: retain validated latent "
            "mapping and derive marker projections from its dose contrasts without replacing z_in. "
            "Counts: 235 Python runs, 23 forwards.\n")
```