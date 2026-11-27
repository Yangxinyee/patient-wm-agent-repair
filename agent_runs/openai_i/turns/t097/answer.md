The architecture confirms that dose effects must propagate through the shared recurrent state. Earlier record analyses support opposite effects from the misleading raw associations: insulin lowers glucose, while norepinephrine raises MAP. I’ll now inspect the retained reconstruction helpers and persistence estimates. These will determine whether a shared-state correction is feasible without damaging factual predictions.

```python
import re
from pathlib import Path

print("=== RECONSTRUCTION HELPERS ===")
helper89 = []
for n89, o89 in list(globals().items()):
    if callable(o89) and hasattr(o89, "__code__"):
        if any(s89 in n89.lower() for s89 in
               ("atom", "pool", "encod", "post", "context", "step", "head", "flow",
                "norm", "act", "reconstruct", "bundle", "transition")):
            helper89.append((n89, o89.__code__.co_varnames[:o89.__code__.co_argcount]))
print(helper89)

print("\n=== KEY TENSORS ===")
for n89 in NAMES:
    if (n89.startswith(("encode.", "encode_action.", "core.z_post", "core.z_in",
                       "core.prelude.", "attr_heads.value_bin"))
        or "posterior" in n89):
        print(n89, SHAPES[n89])

print("\n=== RETAINED OBJECT INVENTORY ===")
inventory89 = []
for n89, o89 in list(globals().items()):
    if n89.startswith("_"):
        continue
    if any(s89 in n89.lower() for s89 in
           ("panel", "state", "bundle", "impulse", "effect", "factual",
            "dose", "target", "sid", "dev", "valid", "train")):
        if hasattr(o89, "shape"):
            inventory89.append((n89, type(o89).__name__, tuple(o89.shape)))
        elif isinstance(o89, (list, dict, tuple)):
            inventory89.append((n89, type(o89).__name__, len(o89)))
print(str(inventory89)[:2500])

print("\n=== PERSISTENCE / IDENTIFICATION EXCERPTS ===")
text89 = Path("/work/notes.md").read_text()
for match89 in list(re.finditer(r"(?im)^.*(?:impulse|persistence|distributed.lag).*$", text89))[:6]:
    start89 = max(0, match89.start() - 100)
    print(text89[start89:match89.end() + 550])

print("\n=== FORWARD ARTIFACTS ===")
for dir89 in sorted(Path("/work").glob("fwd_*")):
    print(dir89.name, [p89.name for p89 in dir89.iterdir()])

with open("/work/notes.md", "a") as f89:
    f89.write(
        "\n## Python 89; forward 6 — select the recurrent edit pathway\n"
        "- Architecture uses shared recurrent state and same-code last-value registers.\n"
        "- Earlier record analyses support proximal insulin effects near -2 and norepinephrine near +3 per rate unit.\n"
        "- Inspecting retained helpers, intervention persistence, and artifacts; no weights changed.\n"
    )
```