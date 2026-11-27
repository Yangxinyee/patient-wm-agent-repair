The architecture confirms that dose information must survive recurrent updates before reaching the value head. Earlier record analyses estimated nonzero effects, but the saved estimates remain conditional on their identification assumptions. I’ll recover the replay and calibration objects with shorter output. Proportional transport under doubling and halving would refute transport as the main source of asymmetry.

```python
import re as _r194_re
import inspect as _r194_inspect

_r194_items = list(globals().items())

print("=== RECORD-ANALYSIS FOLLOW-UP ===")
_r194_text = _r193_Path("/work/notes.md").read_text()
_r194_sections = _r194_text.split("\n## ")
for _r194_section in _r194_sections:
    if any(_r194_word in _r194_section.lower() for _r194_word in
           ("dose strata", "identification", "robustness", "nonlinear", "dynamic target",
            "record-derived", "negative control", "instrument")):
        if not any(_r194_word in _r194_section.lower() for _r194_word in
                   ("restart", "runs 153", "runs 156", "forward 16", "run 135")):
            print(_r194_section[:1100])
            if sum(len(s) for s in _r194_sections[:_r194_sections.index(_r194_section)]) > 17000:
                break

print("=== USER-DEFINED FUNCTION NAMES ===")
_r194_functions = [
    (_r194_n, _r194_o) for _r194_n, _r194_o in _r194_items
    if _r194_inspect.isfunction(_r194_o)
    and getattr(_r194_o, "__module__", "") == "__main__"
]
print("Count:", len(_r194_functions))
for _r194_n, _r194_o in _r194_functions[-65:]:
    print(_r194_n, str(_r194_inspect.signature(_r194_o)))

print("=== RECENT OBJECTS AND HEAD SPECS ===")
_r194_recent = []
for _r194_n, _r194_o in _r194_items:
    if (
        _r194_re.search(r"(184|185|186|187|188|189|190|191|149)(?:_|$)", _r194_n)
        or _r194_n in ("s4", "candidate_e184", "panel", "panel_tune")
    ):
        if _r194_inspect.isfunction(_r194_o):
            continue
        if hasattr(_r194_o, "shape"):
            _r194_desc = f"{type(_r194_o).__name__} {tuple(_r194_o.shape)}"
        elif isinstance(_r194_o, dict):
            _r194_desc = "dict keys " + repr(list(_r194_o)[:18])
        elif isinstance(_r194_o, (list, tuple)):
            _r194_desc = f"{type(_r194_o).__name__} len={len(_r194_o)} " + repr(_r194_o[:1])[:180]
        elif isinstance(_r194_o, str):
            _r194_desc = repr(_r194_o[:220])
        else:
            _r194_desc = repr(_r194_o)[:180]
        _r194_recent.append((_r194_n, _r194_desc))
for _r194_n, _r194_desc in _r194_recent[-65:]:
    print(_r194_n, _r194_desc)

print("=== SOURCE-STRING CANDIDATES ===")
for _r194_n, _r194_o in _r194_items:
    if isinstance(_r194_o, str) and len(_r194_o) > 300 and (
        "def " in _r194_o or "exec(" in _r194_o
    ):
        print(_r194_n, "length", len(_r194_o), repr(_r194_o[:100]))
```