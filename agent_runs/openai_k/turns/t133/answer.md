The retained work confirms that the recurrent-state reconstruction is accurate, but the local readout is not. It also found that head-only edits were too weak and prepared a dose-centered state-token correction. I’ll inspect that candidate’s tensors and diagnostics now, then test it externally; the key question is whether its apparent local preservation survives the official factual evaluation.

```python
import pprint
print("=== LATEST CANDIDATE NOTES ===")
for run124 in [78, 80, 81, 82, 88, 90]:
    matches124 = [s for s in sections123 if re.match(rf"## Runs? {run124}(?:\D|$)", s)]
    for s in matches124:
        print(s[:700])
print("\n=== TARGETS AND DIAGNOSTICS ===")
for k in ["target56", "effects63", "target88", "baseline_slopes89", "diagnostics90"]:
    if k in globals():
        print(k, str(globals()[k])[:1800])
print("\n=== CANDIDATE STRUCTURE ===")
for a, b in globals().get("candidates90", {}).items():
    print("scale", a, "type", type(b).__name__)
    if isinstance(b, dict):
        print([(k, tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
               for k, v in b.items()])
    elif isinstance(b, (list, tuple)):
        print([(type(v).__name__, tuple(v.shape) if hasattr(v, "shape") else str(v)[:80])
               for v in b])
    else:
        print(str(b)[:250])
print("\n=== RECENT RETAINED OBJECTS ===")
for k, v in list(globals().items()):
    if re.search(r"(87|88|89|90)$", k) and not callable(v):
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "dict keys", list(v.keys())[:8])
        elif isinstance(v, (list, tuple)):
            print(k, type(v).__name__, len(v))
print("\n=== RESPONSE-ROUTE WEIGHTS ===")
for n in NAMES:
    if n.startswith(("core.z_post", "core.z_in", "core.z_head")) or "attr_heads.value" in n:
        print(n, SHAPES[n])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 124: token-candidate inspection\n"
            "- Exact recurrent reconstruction is available, but local value readout remains untrusted.\n"
            "- Recovering candidate tensors and diagnostics for an official factual test rather than more readout searches.\n"
            "- No submitted edits. Python runs: 124; forward passes: 4.\n")
```
