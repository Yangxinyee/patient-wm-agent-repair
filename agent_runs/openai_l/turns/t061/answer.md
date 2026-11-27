The recovered evidence supports an insulin response near −2 per rate unit and a norepinephrine response near +3.1, with different time courses. These are record estimates, not yet edit targets with guaranteed rollout accuracy. I’ll now inspect only the reconstruction inputs and functions. A bundle-type breakdown should tell us whether the remaining error affects the action pathway we need to edit.

```python
import dis

print("=== COMPACT RECONSTRUCTION OBJECTS ===")
for nm55 in sorted(list(globals())):
    if any(s in nm55 for s in ["51", "52", "r32_best", "state_train_ids36"]):
        ob55 = globals()[nm55]
        if isinstance(ob55, types.FunctionType):
            continue
        if isinstance(ob55, dict):
            ds55 = "keys=" + repr(list(ob55)[:8])
        elif isinstance(ob55, (list, tuple)):
            ds55 = "len=" + str(len(ob55)) + " elements=" + repr([
                (type(v).__name__, str(getattr(v, "shape", "")))
                for v in ob55[:8]
            ])
        elif hasattr(ob55, "shape"):
            ds55 = "shape=" + str(ob55.shape)
        elif isinstance(ob55, (str, int, float)):
            ds55 = repr(ob55)[:80]
        else:
            ds55 = ""
        print(nm55, type(ob55).__name__, ds55)

print("\n=== VOCAB STRUCTURE ===")
print("keys:", list(VOCAB))
for ky55 in ["codes", "config", "normalization", "normalisation", "categorical"]:
    if ky55 in VOCAB:
        print(ky55, repr(VOCAB[ky55])[:1800])

print("\n=== ESSENTIAL FUNCTION IMPLEMENTATIONS ===")
for nm55 in ["observation_embeddings51", "evaluate_observations51",
             "posterior50", "contexts37"]:
    f55 = globals()[nm55]
    print("\nFUNCTION", nm55, "constants=", f55.__code__.co_consts)
    try:
        print(inspect.getsource(f55))
    except (OSError, TypeError):
        for ins55 in dis.get_instructions(f55):
            print(ins55.opname, ins55.argrepr)
```
