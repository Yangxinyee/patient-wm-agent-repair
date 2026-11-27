The kernels show insulin’s response peaking in the second hour and then declining, while norepinephrine decays faster. A four-state correction could represent these dynamics. I’ll inspect the exact state-update and weight-access helpers so the candidate can be built and undone safely. Its first test will be numerical reconstruction and controlled dose response—not a rollout score alone.

```python
import dis, numpy as np, torch

def describe_code130(f, limit):
    print("\nFUNCTION", f.__name__)
    print("names:", f.__code__.co_names)
    print("constants:", repr(f.__code__.co_consts)[:600])
    text = "\n".join(f"{i.opname} {i.argrepr}" for i in dis.get_instructions(f))
    print(text[:limit])

describe_code130(w63, 700)
describe_code130(transition64, 4600)
print("\nPREPARATION GLOBALS:", prepare103.__code__.co_names)
print("PRELUDE GLOBALS:", prelude_features102.__code__.co_names)
print("PRELUDE CONSTANTS:", prelude_features102.__code__.co_consts)
print("\nATTENTION EXTRACTION TAIL:")
text130 = "\n".join(f"{i.opname} {i.argrepr}"
                    for i in dis.get_instructions(attention_features116))
print(text130[-1600:])

print("\n=== COMPONENT STRUCTURE ===")
print("Bdose125 norms:", np.linalg.norm(Bdose125, axis=0))
print("Bdose125 last row:", Bdose125[-1])
print("pair125:", [(type(x).__name__, getattr(x, "shape", None)) for x in pair125])
print("raw125 RMS:", float(np.sqrt(np.mean(np.asarray(raw125)**2))))
print("raw125 slot RMS:", np.sqrt(np.mean(np.asarray(raw125)**2, axis=(0,2))))

print("\n=== INPUT AND FEATURE VARIABLE NAMES ===")
for suffix in ("60", "75", "81", "102", "116", "118"):
    entries = []
    for k, v in sorted(dict(globals()).items()):
        if k.endswith(suffix) and isinstance(v, (np.ndarray, torch.Tensor, dict)):
            entries.append(k + ":" + str(tuple(v.shape) if hasattr(v, "shape")
                                        else list(v)[:6]))
    print(suffix, " ".join(entries)[:950])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 130 / compact recovery\n"
            "Recovered accepted immediate effects I≈-1.95 and NE≈+3.1, "
            "insulin supported by independent RD; rejected SSM targets are not used. "
            "Eight-lag kernels agree approximately across sensors and reject simple AR1. "
            "Provisional hour-12 targets remain -12.77,+6.38,+4.50,-2.25. "
            "Inspecting exact interfaces to assemble a reversible four-state correction. "
            "No edits saved; forward passes 4/50.\n")
```
