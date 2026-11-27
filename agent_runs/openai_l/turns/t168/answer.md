The inspection stopped on a NumPy-versus-Torch formatting error; it did not change any weights. I’ll fix that and recover the remaining helper signatures and kernel parameters. Then I can test whether the assembled edit produces the intended dose response rather than relying only on matrix-level fits.

```python
def brief156(obj):
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        n = obj.size if isinstance(obj, np.ndarray) else obj.numel()
        return f"shape={tuple(obj.shape)}" + (
            " values=" + repr(obj)[:450] if n <= 24 else "")
    if isinstance(obj, dict):
        return "dict " + repr(list(obj))[:350]
    if isinstance(obj, (list, tuple)):
        return f"{type(obj).__name__} length={len(obj)} " + repr(obj)[:250]
    return repr(obj)[:250]

g156 = dict(globals())
print("=== CONTROLLER AND WIRE PARAMETERS ===")
for name in ["systems152", "poles153", "wire_slopes153", "chosen125",
             "routes145", "controllers153"]:
    obj = g156.get(name)
    print(name, brief156(obj))
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(" ", key, brief156(value))
            if isinstance(value, dict):
                for k, v in value.items():
                    print("    ", k, brief156(v))

print("\n=== CAUSAL / READOUT VARIABLES ===")
for name, obj in sorted(g156.items()):
    if any(s in name.lower() for s in ["kernel", "pole", "readout", "target"]):
        print(name, brief156(obj))

print("\n=== LOCAL HELPER SIGNATURES ===")
for name, obj in sorted(g156.items()):
    if inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            print(name, inspect.signature(obj))
        except Exception:
            pass

print("\n=== CAUSAL NOTE EXCERPTS ===")
lines156 = Path("/work/notes.md").read_text().splitlines()
hits156 = [i for i, line in enumerate(lines156)
           if any(s in line.lower() for s in
                  ["eight-lag", "discontinuity", "accepted immediate",
                   "provisional hour", "original rollout"])]
for i in hits156[-8:]:
    print(textwrap.shorten(
        "\n".join(lines156[max(0, i-1):min(len(lines156), i+2)]),
        width=500, placeholder=" ..."))
```
