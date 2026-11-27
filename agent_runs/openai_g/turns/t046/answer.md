The architecture supports isolating dose magnitude while preserving drug identity and timing. The initial insulin estimates are near zero, but the MAP regressions disagree, and the output was truncated. I’ll recover the nonlinear and timing checks and audit the weights. Consistent, properly timed estimates would support keeping the edit; a reproducible nonzero effect would challenge it.

```python
import numpy as np
from pathlib import Path

p = Path("/work/notes.md")
text = p.read_text()
start = text.find("## Nonlinear")
print("Notebook characters:", len(text))
print("\n=== NONLINEAR AND SUBSEQUENT EVIDENCE ===")
print(text[start:start + 6500])
print("\n=== NOTEBOOK SECTION HEADINGS ===")
print("\n".join(line for line in text.splitlines() if line.startswith("## ")))

print("\n=== WEIGHT AUDIT ===")
audit = []
for name in edited():
    current = np.asarray(load(name))
    original = np.asarray(load_original(name))
    item = {
        "name": name,
        "shape": list(current.shape),
        "finite": bool(np.isfinite(current).all()),
        "change_norm": float(np.linalg.norm(current - original)),
    }
    if name in ("embed.bins.dose.weight", "embed.bins.rate.weight"):
        item["absent_row_unchanged"] = bool(np.array_equal(current[0], original[0]))
        item["present_rows_identical"] = bool(np.array_equal(
            current[1:], np.broadcast_to(current[1], current[1:].shape)))
    elif name == "embed.numeric.weight":
        item["dose_and_rate_columns_zero"] = bool(np.all(current[:, [2, 4]] == 0))
        item["other_columns_unchanged"] = bool(np.array_equal(
            current[:, [0, 1, 3, 5]], original[:, [0, 1, 3, 5]]))
    audit.append(item)
print(audit)

p.write_text(text + "\n## Recovery review, Python run 37\n"
             "Recovering truncated nonlinear and timing evidence. No weights changed.\n"
             "Weight audit: " + repr(audit) + "\n")
```