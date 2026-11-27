The recovered estimates suggest effects lasting several hours, and the withheld-assay checks did not reveal a clear residual association. That supports—but does not prove—the causal interpretation. I’ll now inspect only the retained edit candidates and reconstruction helpers. The candidate must preserve factual states and represent proportional dose changes before a rollout test is worthwhile.

```python
def summary69(obj):
    if hasattr(obj, "shape"):
        return f"{type(obj).__name__}{tuple(obj.shape)}"
    if isinstance(obj, dict):
        return "dict{" + ", ".join(
            f"{str(k)[:45]}: {summary69(v)}"
            for k, v in list(obj.items())[:5]) + "}"
    if isinstance(obj, (list, tuple)):
        return f"{type(obj).__name__}[{len(obj)}] " + ", ".join(
            summary69(v) for v in obj[:3])
    return repr(obj)[:100]

print("=== TRACE / OFFSET / LATENT CANDIDATES ===")
for name in sorted(list(globals())):
    if name.startswith(("rr50", "rr51", "rr58")):
        print(name, summary69(globals()[name]))

print("\n=== USER-DEFINED HELPERS ===")
for name in sorted(list(globals())):
    obj = globals()[name]
    if inspect.isfunction(obj) and name.startswith(("r", "cpu", "make", "fit", "encode")):
        print(name, str(inspect.signature(obj)),
              "uses:", ",".join(obj.__code__.co_names[:12]))

print("\n=== SELECTED TENSOR SHAPES ===")
for name in NAMES:
    if any(s in name for s in ("z_post", "z_in", "encode_action", "attr_heads.value")):
        print(name, SHAPES[name])

print("\n=== VOCAB STRUCTURE ===")
print(list(VOCAB.keys()))
print("config", VOCAB.get("config"))
print("edits", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 69\n"
            "Recovered six-hour kernels: insulin [-1.850,-2.256,-1.943,-1.561,-1.022,-0.690]; "
            "NE [3.351,2.621,1.717,0.943,0.494,0.409]. "
            "Conditional-exogeneity assumptions remain necessary. "
            "Inspecting retained candidates without repeated evidence dumps. No edits.\n"
            "Counts: 69 Python runs, 6 forward passes.\n")
```