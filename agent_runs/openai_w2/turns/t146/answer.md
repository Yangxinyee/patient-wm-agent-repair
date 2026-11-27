The architecture confirms that a recurrent-input edit can affect future readings without changing the output heads. The earlier record analysis also found treatment-linked changes in measurement availability, so I won’t treat the initial threshold estimates as definitive. The output was truncated before the recent variables; I’ll retrieve those compactly before constructing the candidate.

```python
import re

print("=== NOTEBOOK SIZE AND EARLY CONCLUSIONS ===")
print("Characters:", len(notes_recover))
early_notes = notes_recover[:-8000]
for section in re.split(r"(?=^## )", early_notes, flags=re.M):
    lines = section.strip().splitlines()
    if lines and not any('"' in s for s in lines[:4]):
        text = "\n".join(lines)
        if any(s in text.lower() for s in
               ["target", "dynamics", "causal", "persistence", "run 8",
                "run 9", "run 10", "run 11", "run 12", "run 13",
                "run 14", "run 15", "run 16", "run 17", "run 18",
                "run 19", "run 2", "run 3"]):
            print(text[:550])

print("\n=== RECENT KERNEL OBJECTS ===")
for k in sorted(list(globals())):
    if re.search(r"(11[0-9]|12[0-8])", k) or any(
        s in k.lower() for s in ["repair_head", "prelude", "steer", "benchmark", "target"]
    ):
        v = globals()[k]
        desc = list(v.keys())[:12] if isinstance(v, dict) else ""
        print(k, type(v).__name__, getattr(v, "shape", None), desc)

print("\n=== HELPER FUNCTION NAMES ===")
print(" ".join(sorted(k for k, v in list(globals().items())
                     if callable(v) and getattr(v, "__module__", "") == "__main__")))
print("\nCurrent edits:", edited())
```