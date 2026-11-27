The retained edits have very large weight changes, and the rollout error barely responds to stronger local constraints. Off-record state behavior is therefore worth testing, though it is not yet an established cause. I’ll recover the exact carrier and regression helpers so the next edit can address generated-history robustness without sacrificing factual fit.

```python
import dis, io
print("REGRESSION AND CARRIER VARIABLES")
for suffix in ("154", "206", "233", "255", "261", "262", "267"):
    items = []
    for k, v in list(globals().items()):
        if k.endswith(suffix):
            shape = getattr(v, "shape", None)
            desc = list(v.keys())[:12] if isinstance(v, dict) else shape
            if isinstance(v, (int, float, str)):
                desc = str(v)[:150]
            items.append((k, type(v).__name__, desc))
    print(suffix, items)

print("\nHELPER IMPLEMENTATIONS")
_budget288 = 6500
for name in ("latent206", "posterior261", "parts254", "normalize267", "original_tokens278"):
    fn = globals()[name]
    try:
        text = inspect.getsource(fn)
    except Exception:
        text = dis.Bytecode(fn).dis()
    text = text[:min(1700, _budget288)]
    print("\n", name, "\n", text)
    _budget288 -= len(text)
    if _budget288 <= 0:
        break

_notes288 = open("/work/notes.md").read()
print("\nCARRIER CONSTRUCTION NOTES")
for runno in (154, 155, 206, 222, 267):
    match = re.search(r"Python run[s]?:?\s*" + str(runno) + r"\b", _notes288)
    if match:
        print(_notes288[match.start():match.start()+500])
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 288; forwards 28. Recovering exact dose-carrier and total-response regression helpers to examine off-record robustness. No weights changed; checkpoint287 remains the gamma=10 candidate.\n")
```