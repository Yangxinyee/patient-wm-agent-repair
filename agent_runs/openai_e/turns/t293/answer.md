The saved nonlinear detector has the expected shape, and the architecture shows how it could alter the prelude through a low-rank FFN edit. Its MAP response is not yet established. I’ll check the earlier direction tests and available contexts so the next trial uses a measured direction rather than an arbitrary one.

```python
import re, dis, numpy as np, torch

text253 = open("/work/notes.md").read()
sections253 = re.split(r"(?=^## )", text253, flags=re.M)
print("PRIOR DIRECTION TESTS")
for n in [93, 94, 95, 96, 134, 135, 136, 137, 148, 151]:
    matches = [s for s in sections253
               if re.search(r"\bRun[s]?\s+" + str(n) + r"\b", s[:130])]
    for s in matches:
        print(s[:850])

print("\nDIRECTION AND CONTEXT VARIABLES")
for k, v in list(globals().items()):
    wanted = (
        re.search(r"(45|46|93|94|95|96|135|136|137|191)$", k)
        or any(s in k.lower() for s in ["map_direction", "map_dir", "output_dir"])
    )
    if not wanted:
        continue
    if isinstance(v, (np.ndarray, torch.Tensor)):
        print(k, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "dict", list(v)[:12])
    elif callable(v) and hasattr(v, "__code__"):
        print("FUNCTION", k,
              v.__code__.co_varnames[:v.__code__.co_argcount],
              "names", v.__code__.co_names)

print("\nPRELUDE HIDDEN IMPLEMENTATION")
dis.dis(prelude_hidden93)

print("\nWEIGHT DICTIONARY VERSION")
for dn in ["W0", "CW", "core_W", "W46"]:
    dd = globals().get(dn)
    if isinstance(dd, dict) and "core.b_in.weight" in dd:
        aa = dd["core.b_in.weight"]
        aa = aa.detach().cpu().numpy() if isinstance(aa, torch.Tensor) else np.asarray(aa)
        print(dn, "b_in error vs original",
              float(np.max(np.abs(aa - load_original("core.b_in.weight")))),
              "vs current", float(np.max(np.abs(aa - load("core.b_in.weight")))))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 253: recover measured prelude direction\n"
            "Nonlinear detector has 768 raw GELU coefficients plus intercept. "
            "Inspecting earlier nonlinear response tests and anchored contexts "
            "to reuse a measured output direction. Joint236 unchanged.\n"
            "Python runs: 253; forwards: 29.\n")
```