The architecture confirms a direct correction point: `core.b_in` maps the dose-sensitive prelude output into recurrent state. I’ll inspect the validated replay helpers and recover the dose-channel construction. This route is promising only if changed doses activate features that vary little on factual records; otherwise amplification would likely damage factual fit.

```python
import inspect, dis, re

print("=== HELPER GLOBAL DEPENDENCIES ===")
for r233_name in ("r59_core", "core228", "posterior_token154", "r179_capture_head", "r186_heads"):
    r233_fn = globals()[r233_name]
    print(r233_name, "args", r233_fn.__code__.co_varnames[:r233_fn.__code__.co_argcount],
          "globals", r233_fn.__code__.co_names)
    try:
        r233_src = inspect.getsource(r233_fn)
        print(r233_src[:8000])
    except (OSError, TypeError):
        pass

print("\n=== RETAINED SOURCE STRINGS ===")
for r233_name, r233_obj in list(globals().items()):
    if isinstance(r233_obj, str) and "def " in r233_obj and any(
        s in r233_obj for s in ("def core228", "def r59_core", "def r179_capture_head", "def r186_heads")
    ):
        print(r233_name, len(r233_obj))
        print(r233_obj[:13000])

print("\n=== RECENT REPLAY OBJECTS ===")
for r233_name, r233_obj in list(globals().items()):
    if r233_name.startswith(("r228", "r229", "r230", "r231", "ctx228", "i228", "i230")):
        if isinstance(r233_obj, dict):
            print(r233_name, "dict", list(r233_obj)[:20])
        elif hasattr(r233_obj, "shape"):
            print(r233_name, type(r233_obj).__name__, tuple(r233_obj.shape))
        elif isinstance(r233_obj, (list, tuple)):
            print(r233_name, type(r233_obj).__name__, len(r233_obj))
        elif isinstance(r233_obj, (float, int, str)) and len(str(r233_obj)) < 180:
            print(r233_name, r233_obj)

r233_notes = Path("/work/notes.md").read_text()
r233_sections = re.split(r"\n(?=## )", r233_notes)
print("\n=== EARLIER IDENTIFICATION AND CHANNEL EVIDENCE ===")
r233_selected = [
    s for s in r233_sections
    if any(k in s.lower() for k in ("randomiz", "six-lag", "causal coefficient", "physical dose",
                                     "reserved neuron", "candidate158"))
]
for r233_s in r233_selected[:6] + r233_selected[-5:]:
    print(r233_s[:1800])
with open("/work/notes.md", "a") as r233_f:
    r233_f.write("\n## Run 233: inspect validated helper interfaces\n"
                 "- core.b_in is an available downstream dose-routing boundary; factual selectivity remains unproven.\n"
                 "- Recovering helper dependencies and record-identification evidence.\n"
                 "- No weights changed. Python runs: 233; forward passes: 17.\n")
```
