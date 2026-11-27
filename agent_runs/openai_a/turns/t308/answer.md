The architecture confirms that the current edit changes only the reading head. The recovered record analyses support a negative insulin effect, while norepinephrine estimates vary with adjustment. The output was truncated before the calibration helpers appeared, so I’ll recover those separately; this will determine whether a further targeted edit is feasible.

```python
import inspect, json, re
print("VOCAB keys:", list(VOCAB))
for _key in ("numeric", "numeric_attributes", "normalisation", "value_bin_centres"):
    if _key in VOCAB:
        _v = VOCAB[_key]
        print(_key, type(_v).__name__)
        if isinstance(_v, dict):
            print("keys:", list(_v.keys())[:15])
            for _ek, _ev in _v.items():
                if str(_ek) in ("6", "glucose_finger"):
                    print("finger entry:", _ek, _ev)
        elif isinstance(_v, list):
            for _entry in _v:
                if isinstance(_entry, dict) and _entry.get("code") == "glucose_finger":
                    print("finger entry:", _entry)

_funcs = [_n for _n, _o in list(globals().items())
          if inspect.isfunction(_o) and getattr(_o, "__module__", "") == "__main__"]
print("HELPER NAMES:", ", ".join(sorted(_funcs)))

print("CHECKPOINTS AND CACHES:")
for _n, _o in sorted(list(globals().items())):
    if not _n.startswith("_") and any(s in _n.lower() for s in ("checkpoint", "cache", "calib")):
        if isinstance(_o, dict):
            print(_n, "dict keys:", list(_o.keys())[:12])
        elif hasattr(_o, "shape"):
            print(_n, type(_o).__name__, tuple(_o.shape))
        elif isinstance(_o, (str, int, float)):
            print(_n, repr(_o)[:150])

print("RECENT NONPRIVATE VARIABLE NAMES:")
print(", ".join([_n for _n in list(globals()) if not _n.startswith("_")][-180:]))

_notes = Path("/work/notes.md").read_text()
print("IDENTIFICATION SUMMARY EXCERPTS:")
for _m in list(re.finditer(r"(?im)^## .*(?:fuzzy|identification|persistence|causal|norepinephrine).*", _notes))[-5:]:
    print(_notes[_m.start():min(_m.start()+650, len(_notes))])
with open("/work/notes.md", "a") as _f:
    _f.write("\n## Restart recovery; Python 264, forwards 32/50\n"
             "- Python 262 bin-key lookup failed; no weights changed.\n"
             "- Re-read architecture and record columns. Current edit remains gated8.\n"
             "- Recovering helpers in bounded output after Python 263 was truncated.\n")
```
