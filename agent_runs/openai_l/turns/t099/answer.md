The architecture confirms that actions affect both the latent update and recurrence. Since the posterior matches, I’ll check the memory inputs—especially mask polarity. Recovering the earlier near-zero state error would validate local dose probes; otherwise I’ll keep rejecting them.

```python
import dis, re

def show91(f, limit=3000):
    try:
        text = inspect.getsource(f)
    except Exception:
        text = "\n".join(
            f"{i.opname} {i.argrepr}" for i in dis.Bytecode(f)
        )
    print(text[:limit])

for name in ("transition64", "bundle_inputs74", "latent_probe88"):
    print("\n=== " + name + " ===")
    show91(globals()[name], 2800)

print("\n=== RECENT VARIABLES ===")
for name, value in list(globals().items()):
    if re.search(r"(76|80|81|82|83|85|88|89)$", name):
        shape = getattr(value, "shape", None)
        if shape is not None:
            print(name, tuple(shape))
        elif isinstance(value, dict):
            print(name, "keys", list(value)[:8])
        elif isinstance(value, (str, int, float, bool)):
            print(name, str(value)[:160])

print("\n=== BUNDLE INPUT RETURN STRUCTURE ===")
inputs91 = bundle_inputs74(valid_indices51[:8])
if isinstance(inputs91, dict):
    iterable91 = inputs91.items()
else:
    iterable91 = enumerate(inputs91)
for key, value in iterable91:
    print(key, type(value).__name__, getattr(value, "shape", None))
    if getattr(value, "dtype", None) == torch.bool:
        print("  true count:", int(value.sum()), "total:", value.numel())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 91 / targeted memory audit\n"
            "Architecture reviewed. Posterior matches, so checking "
            "transition inputs and mask polarity rather than interpreting "
            "the invalid latent probe. No tensors edited.\n")
```