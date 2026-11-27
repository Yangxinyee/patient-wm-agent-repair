The records support effects substantially larger than the original model’s rollouts, but those causal estimates remain separate from whether an edit can deliver them. I’ll now inspect the exact state-update operations and retained response kernels. A useful route must carry dose deviations forward without turning ordinary treatment–condition associations into effects.

```python
import dis

print("=== RELEVANT WEIGHTS ===")
for n121 in NAMES:
    if any(s121 in n121 for s121 in
           ("core.c_out", "core.b_in", "core.coda", "core.prelude",
            "core.a_in", "core.delta_in", "attr_heads.value_bin")):
        print(n121, SHAPES[n121])

print("\n=== HELPER DEPENDENCIES ===")
for n121 in ("w63", "linear63", "norm63", "transition64", "layer63",
             "attention_features116", "prepare103", "head_inputs96"):
    f121 = globals()[n121]
    print(n121, "names:", f121.__code__.co_names)
    print("constants:", [x for x in f121.__code__.co_consts
                         if isinstance(x, (str, int, float))][:35])

print("\n=== TRANSITION BYTECODE ===")
s121 = "\n".join(f"{x.opname} {x.argrepr}"
                 for x in dis.get_instructions(transition64))
print(s121[:3400])
print("\n=== HEAD-INPUT BYTECODE ===")
s121 = "\n".join(f"{x.opname} {x.argrepr}"
                 for x in dis.get_instructions(head_inputs96))
print(s121[:1800])

print("\n=== RETAINED CAUSAL KERNELS ===")
if "kernels19" in globals():
    for k121, v121 in kernels19.items():
        if isinstance(v121, dict):
            print(k121, {k: (np.asarray(v).round(4).tolist()
                            if isinstance(v, (np.ndarray, list, tuple)) else str(v)[:300])
                         for k, v in v121.items()})
        else:
            print(k121, str(v121)[:1500])

print("\n=== STATE/DATA OBJECT NAMES ===")
print(" ".join(k for k in globals()
               if any(s in k.lower() for s in
                      ("bundle", "allstate", "state", "memory", "causal", "kernel",
                       "target", "train", "tuning", "cache", "weight"))
               and not k.startswith("_"))[:1800])
```