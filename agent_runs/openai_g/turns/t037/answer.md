The latest test increased factual NLL by just 0.00012 nats, well below the 0.005 limit. That supports retaining the edit, but the isolated late norepinephrine signal remains unresolved. I’ll recover the earlier evidence and validation coverage; consistent null effects would support dose invariance, while a reproducible dose response would challenge it.

```python
from pathlib import Path

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== DATA README ===")
print(Path("/data/README.md").read_text())

notes_text = Path("/work/notes.md").read_text()
print("=== NOTEBOOK LENGTH ===", len(notes_text))
print("=== NOTEBOOK SECTIONS ===")
print("\n".join(line for line in notes_text.splitlines() if line.startswith("#")))
print("=== EARLIER NOTEBOOK ===")
print(notes_text[:15000])

print("=== EXISTING KERNEL VARIABLES ===")
print(sorted(k for k in globals() if not k.startswith("_")))
print("=== FORWARD OUTPUT FILES ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, sorted(p.name for p in folder.iterdir()))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Restart recovery, run 30\n"
        "Latest paired factual test: 100 tuning stays, 4,838 readings; "
        "original NLL 1.51558, edited NLL 1.51570, difference +0.00012. "
        "No weights changed. Recovering prior evidence and exact validation "
        "coverage before completing remaining factual and causal checks.\n"
    )
```