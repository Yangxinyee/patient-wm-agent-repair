The architecture confirms a 768-feature GELU layer in the prelude, which could provide a richer detector than the current edit. The output was truncated before the saved-data diagnostics, so I’ll retrieve those compactly. This route is worth pursuing only if earlier tests did not already rule it out.

```python
from pathlib import Path
import numpy as np, re, dis

print("HELPER CONTRACTS")
for name in ("prelude_hidden93", "prelude46", "core_ffn", "core_lin"):
    fn = globals().get(name)
    if callable(fn) and hasattr(fn, "__code__"):
        c = fn.__code__
        print(name, "actual_name=", fn.__name__,
              "locals=", c.co_varnames,
              "names=", c.co_names,
              "constants=", [x for x in c.co_consts
                             if isinstance(x, (str, int, float))])
        if name == "prelude_hidden93":
            print("return-tail:", [(i.opname, i.argrepr)
                                   for i in list(dis.get_instructions(fn))[-26:]])

print("\nRETAINED FEATURE ARRAYS")
suffix239 = re.compile(r"(46|47|56|57|81|82|93|96|101|148|151|154|155|199|236)$")
items239 = [
    f"{name}:{type(value).__name__}{tuple(value.shape)}"
    for name, value in list(globals().items())
    if suffix239.search(name) and hasattr(value, "shape")
]
text239 = "\n".join(items239)
print(text239[:3000])
if len(text239) > 3000:
    print("remaining names:", [s.split(":")[0] for s in items239
                               if s not in text239[:3000]])

print("\nSAVED CANDIDATE CONTRACTS")
for filename in ("joint236.npz", "ne_detector81.npz",
                 "ne_candidate151.npz", "ne_candidate199.npz"):
    with np.load("/work/" + filename, allow_pickle=False) as a:
        print(filename)
        for key in a.files:
            x = a[key]
            print(key, x.shape, str(x.dtype),
                  repr(x.tolist())[:250] if x.size <= 12 else "")

print("\nSELECTED METHOD NOTES")
notes239 = Path("/work/notes.md").read_text()
for section in re.split(r"(?=^## )", notes239, flags=re.M):
    heading = section.splitlines()[0] if section else ""
    if re.search(r"\b(?:93|96|151|199)\b", heading):
        print(section[:1100])

print("\nCURRENT EDITS", edited())
print("b_in shape", SHAPES["core.b_in.weight"])
with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 239: compact recovery after truncated output\n"
        "Architecture confirms a 768-wide GELU prelude FFN. "
        "Recovering helper return contracts, retained arrays and prior NE "
        "method notes with bounded output; joint236 remains unchanged.\n"
        "Python runs: 239; forwards: 29.\n"
    )
```