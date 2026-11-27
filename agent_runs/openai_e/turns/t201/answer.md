The residual equals one additional `last_numeric` projection exactly. That could mean either the rebuilt input is missing a projection or the retained reference counts it twice. I’ll check the probability helpers and their inputs against the GPU output before deciding which is correct.

```python
print("RETAINED HELPER LISTS")
for name in ("funcnames165", "inspectnames165", "funcs164"):
    print(name, globals().get(name))
print("\nSTATIC / PROBABILITY OBJECTS")
for k, v in sorted(list(globals().items())):
    if re.search(r"static|base|with_code|last_feat|prob|logit|query|head_input", k, re.I):
        print(k, type(v).__name__, getattr(v, "shape", ""))
        if callable(v) and hasattr(v, "__code__"):
            print(" ", inspect.signature(v))

print("\nATTRIBUTE/REGISTER FUNCTIONS")
for k, v in sorted(list(globals().items())):
    if callable(v) and hasattr(v, "__code__"):
        strings172 = [x for x in v.__code__.co_consts if isinstance(x, str)]
        if any("last_numeric" in x or "attr_heads.value_bin" in x for x in strings172):
            print(k, inspect.signature(v))
            print(compact_bytecode166(v, budget=1900))

print("\nDIRECT INPUT COMPARISONS")
projection172 = register168 @ W0["last_numeric.weight"].T
for name in ("base162", "d169", "resid169"):
    obj172 = globals().get(name)
    if obj172 is not None:
        arr172 = obj172.detach().cpu().numpy() if torch.is_tensor(obj172) else np.asarray(obj172)
        if arr172.shape == tuple(static168.shape):
            print(name,
                  "RMS", np.sqrt(np.mean(arr172**2)),
                  "minus static RMS", np.sqrt(np.mean((arr172-static168.numpy())**2)),
                  "minus projection RMS", np.sqrt(np.mean((arr172-projection172.numpy())**2)))
print("HISTORY AVAILABLE", [k for k in ("In", "_ih") if k in globals()])
if "In" in globals():
    print("LAST CELL PREFIX", str(In[-2])[:1000])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 172: distinguish missing from duplicated numeric projection\n"
            "Residual equals last_numeric.weight times the three populated register features exactly. "
            "Auditing retained reference and probability helpers against GPU output before choosing assembly. "
            "No weights changed. Python runs: 172; forwards: 22.\n")
```