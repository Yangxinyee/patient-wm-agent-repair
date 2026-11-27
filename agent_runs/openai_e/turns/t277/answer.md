The joint edit passed the latest factual check, but norepinephrine doubling remains almost unrepaired. I’ll inspect the prelude’s nonlinear features and saved calibration data without changing weights. A better detector must distinguish altered doses while staying quiet on factual records; otherwise this route is unlikely to help safely.

```python
from pathlib import Path
import inspect, re, numpy as np, torch, io, contextlib, dis

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

def show_function238(name, limit=6500):
    fn = globals().get(name)
    if not callable(fn):
        return
    print("\nFUNCTION", name, inspect.signature(fn))
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        source = None
        for entry in reversed(globals().get("_ih", [])):
            marker = "def " + name + "("
            if marker in entry:
                start = entry.index(marker)
                remainder = entry[start:]
                stop = re.search(r"\n(?=def |class )", remainder[1:])
                source = remainder if stop is None else remainder[:stop.start() + 1]
                break
        if source is None:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                dis.dis(fn)
            source = buf.getvalue()
    print(source[:limit])

for name in ("prelude_hidden93", "prelude46", "core_ffn"):
    show_function238(name)

print("\nRELEVANT RETAINED ARRAYS")
suffix = re.compile(r"(46|47|56|57|81|82|93|96|101|148|151|154|155|199|236)$")
for name, value in list(globals().items()):
    if suffix.search(name) and hasattr(value, "shape"):
        print(name, type(value).__name__, tuple(value.shape))

print("\nSAVED CANDIDATE CONTRACTS")
for filename in ("joint236.npz", "ne_detector81.npz",
                 "ne_candidate151.npz", "ne_candidate199.npz"):
    with np.load("/work/" + filename, allow_pickle=False) as archive:
        print(filename)
        for key in archive.files:
            value = archive[key]
            print(" ", key, value.shape, value.dtype,
                  value.tolist() if value.size <= 12 else "")

notes238 = Path("/work/notes.md").read_text()
sections238 = re.split(r"(?=^## )", notes238, flags=re.M)
print("\nSELECTED METHOD NOTES")
for section in sections238:
    heading = section.splitlines()[0] if section else ""
    if re.search(r"\b(?:93|96|151|199)\b", heading):
        print(section[:2600])

print("\nCURRENT EDITS", edited())
print("b_in shape", SHAPES["core.b_in.weight"])
with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 238: nonlinear prelude route inspection\n"
        "Joint236 retained unchanged. Latest paired factual cost .00441; "
        "pooled shifts insulin -11.847/+4.000, NE +.115/-.882. "
        "Inspecting architecture, nonlinear prelude helpers, retained feature "
        "arrays and saved NE detector contracts before any new edit. "
        "Hypothesis: a richer detector could increase intervention sensitivity "
        "without increasing factual disturbance; this remains untested.\n"
        "Python runs: 238; forwards: 29.\n"
    )
```