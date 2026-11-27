The later analyses support an insulin response near −2 per unit at the next hour, with a delayed tail, and a norepinephrine response near +3.1 with faster decay. The original rollouts appear too weak for both drugs. I’ll examine the value-head features for a narrow correction; preserving the other heads should limit collateral changes.

```python
import dis, inspect, numpy as np, torch

print("=== RECORD-BASED ROLLOUT TARGETS ===")
for name in ["record_targets22", "record_row_targets25", "kernels22",
             "kernel_se19", "ar2_predictions19"]:
    obj = globals().get(name)
    print(name, repr(obj)[:1800])

print("\n=== EXACT DECODER ===")
for fn in [context64, decode67, register67]:
    print("\nFUNCTION", fn.__name__)
    try:
        print(inspect.getsource(fn))
    except Exception:
        print("arguments:", fn.__code__.co_varnames[:fn.__code__.co_argcount])
        for ins in dis.get_instructions(fn):
            if ins.opname not in ["RESUME", "CACHE", "PRECALL", "PUSH_NULL"]:
                print(ins.opname, ins.argrepr)

print("\n=== RECENT VARIABLE NAMES ===")
recent_names78 = [n for n in globals()
                  if re.search(r"(65|66|67|72|73|74|75)$", n)
                  and not n.startswith("_")]
print(" ".join(recent_names78))
print("\n=== VALUE HEAD TENSORS ===")
for n in NAMES:
    if "attr_heads.value_bin" in n:
        print(n, SHAPES[n])
```