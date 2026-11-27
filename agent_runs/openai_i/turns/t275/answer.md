The sharper selector still produced a +0.10705 factual-NLL increase, effectively unchanged from the rejected candidate. That refutes leakage alone as the explanation. I’ll restore the neutral edit and inspect startup states and attention biases; an uncontrolled startup signal could explain damage before treatment.

```python
for name245 in list(edited()):
    drop_edit(name245)
neutral245 = np.load("/work/neutral209.npz")
for name245 in neutral245.files:
    save_edit(name245, neutral245[name245])

print("=== ATTENTION / INITIALIZATION TENSORS ===")
for name245 in NAMES:
    if (name245.startswith("core.prelude.0")
        and any(s in name245 for s in ["norm", "attn.q", "attn.kv", "attn.out"])):
        print(name245, SHAPES[name245])
    elif any(s in name245.lower() for s in ["h0", "init", "slots"]):
        print(name245, SHAPES[name245])

print("\n=== ALL-BUNDLE REPLAY ARRAYS ===")
for name245 in sorted(list(globals())):
    obj245 = globals()[name245]
    if hasattr(obj245, "shape") and len(obj245.shape) >= 1:
        if obj245.shape[0] == 4035 and not name245.startswith(("a24", "z24")):
            print(name245, tuple(obj245.shape))

print("\n=== CORE HELPER CONTRACTS ===")
for name245 in sorted(list(globals())):
    obj245 = globals()[name245]
    if callable(obj245) and any(
        s in name245 for s in ["call_core", "prelude", "attention", "norm136", "step", "replay_route"]
    ):
        if hasattr(obj245, "__code__"):
            print(name245, str(inspect.signature(obj245)),
                  "uses", obj245.__code__.co_names[:30])

print("\n=== STARTUP INDEX / PRIVATE INNOVATIONS ===")
print("Retained token source indices:", np.asarray(src237)[:12])
print("First 100 innovation max:", np.max(np.abs(array241(innovation237)[:100])))
print("First 100 inactive count:", np.asarray(inactive237[:100], dtype=bool).sum())
print("Current edits:", edited())

with open("/work/notes.md", "a") as f245:
    f245.write("\n## Forward 21; Python 245 — reject sharper candidate\n"
               "- Factual NLL remains 1.73860 versus 1.63155 original (+.10705).\n"
               "- Sharpening attention did not repair factual damage; leakage-only explanation refuted.\n"
               "- Restored neutral209. Investigating startup selection and exact constant-query options.\n")
```
