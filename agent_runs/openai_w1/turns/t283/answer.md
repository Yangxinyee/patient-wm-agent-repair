The factual-safe baseline has been restored. The recovered notes support a glucose-lowering insulin effect and a MAP-raising norepinephrine effect, conditional on adequate adjustment for dose selection. The output was truncated before the later checks and implementation details, so I’ll retrieve those next. Agreement across the robustness checks would strengthen the effect estimates; disagreement would limit how aggressively I edit.

```python
import re
notes250 = Path("/work/notes.md").read_text()
print("EARLY ROBUSTNESS")
for label in ("Run 18", "Run 19", "Run 20"):
    m = re.search(r"(?m)^## "+re.escape(label)+r"\b", notes250)
    if m:
        tail = notes250[m.start():]
        nxt = re.search(r"\n## ", tail[4:])
        sec = tail[:nxt.start()+4] if nxt else tail
        print(sec[:1800])
print("\nLATER NUMERIC IDENTIFICATION")
numeric250 = [
    s for s in notes250.splitlines()
    if any(k in s.lower() for k in ("kernel=", "long-run", "long run", "causal target",
        "dynamic target", "deconvol", "placebo", "withheld-assay", "lag coefficient",
        "impulse response", "effect estimates", "target shift"))
]
for s in numeric250[-20:]:
    print(s[:350])
print("\nIMPLEMENTATION OBJECTS")
for n, v in list(globals().items()):
    if n.startswith("_"):
        continue
    if any(k in n.lower() for k in ("236", "245", "246", "247", "211", "221", "bmean", "causal", "kernel", "dyn")):
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(n, tuple(v.shape))
        elif isinstance(v, dict):
            print(n, "dict", len(v), list(v)[:5])
        elif callable(v) and hasattr(v, "__code__"):
            print(n, "function", v.__code__.co_varnames[:v.__code__.co_argcount])
        elif isinstance(v, (float, int, str)):
            print(n, str(v)[:100])
print("\nFUNCTION NAMES")
print([(n, v.__code__.co_varnames[:v.__code__.co_argcount])
       for n, v in list(globals().items())
       if callable(v) and hasattr(v, "__code__") and not n.startswith("_")][-70:])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 250\nRun249 restored safe236; output truncated after architecture "
            "and early conditional-effect estimates. Narrow recovery of robustness, "
            "dynamic targets and candidate construction. No weights changed. "
            "Counts: 250 Python runs, 24 forwards.\n")
```