The dose/rate-bin amplification is installed but untested. I’ll recover the probe IDs and previous results before testing it. A stronger drug response would support this edit only if factual NLL stays within the 0.005 limit.

```python
from pathlib import Path

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\nRELEVANT KERNEL VARIABLES")
for key, value in sorted(list(globals().items())):
    if key.startswith("_"):
        continue
    if any(term in key.lower() for term in
           ("probe", "ids", "baseline", "gain", "effect", "target", "bin")):
        shape = getattr(value, "shape", None)
        print(key, type(value).__name__, shape)
        if isinstance(value, (str, int, float)):
            print(" ", value)
        elif isinstance(value, (list, tuple)) and len(value) <= 100:
            if all(isinstance(x, (str, int, float)) for x in value):
                print(" ", value)

print("\nNOTEBOOK: EARLY EVIDENCE AND TESTS")
notes = Path("/work/notes.md").read_text()
print(notes[:20000])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Python run 100: restart recovery\n"
            "- Nonzero dose/rate-bin contrast gain 3 remains installed with numeric gain 4.\n"
            "- Recovering architecture, probe IDs, and prior evidence; no edits changed.\n"
            "- Candidate factual safety and rollout benefit remain untested.\n"
            "- Forward passes used: 38; 12 remain.\n")
```