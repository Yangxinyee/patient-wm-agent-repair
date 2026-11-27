The architecture confirms that the current edit changes how action information enters the recurrent state. The output was truncated before showing the saved candidates, so I still cannot safely reconstruct a trial. I’ll inspect only the relevant kernel objects next. The refit remains promising only if direct factual testing improves NLL without losing the intended dose responses.

```python
import re, inspect
g208 = dict(globals())

def describe208(name, value):
    if hasattr(value, "shape"):
        return f"{name}: {type(value).__name__} {tuple(value.shape)}"
    if isinstance(value, dict):
        return f"{name}: dict[{len(value)}] keys={repr(list(value)[:8])[:220]}"
    if callable(value):
        try:
            return f"{name}: function {inspect.signature(value)}"
        except Exception:
            return f"{name}: callable"
    return f"{name}: {type(value).__name__} {repr(value)[:150]}"

recent208 = [
    (k, v) for k, v in sorted(g208.items())
    if re.search(r"(203|204|205|206)", k)
]
print("RECENT OBJECTS")
for k, v in recent208:
    print(describe208(k, v))

print("\nSAVED WEIGHTS / BUILDERS")
for k, v in sorted(g208.items()):
    if (k.startswith(("validated_", "candidate")) or
        (callable(v) and any(s in k.lower() for s in
                            ["fit", "constr", "install", "prelude", "build"]))):
        print(describe208(k, v))

print("\nSELECTED LATE RECORD-EVIDENCE NOTES")
sections208 = notes_text.split("\n## ")
for sec in sections208:
    heading = sec.splitlines()[0]
    if any(s in heading.lower() for s in
           ["178", "181", "182", "183", "causal evidence"]):
        print(sec[:1800])

print("\nCURRENT EDIT", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run208: compact candidate recovery\n"
            "Architecture read; no edits. Run207 output was truncated before "
            "candidate inventory. Inspecting recent saved objects only.\n")
```