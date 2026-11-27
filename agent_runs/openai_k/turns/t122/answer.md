The bin-boundary test barely changed the mismatch, so bin conventions do not explain it. I’ll recover the architecture and earlier identification results before choosing an edit. A useful reconstruction fix must match the external factual predictions, not just improve the local score.

```python
from pathlib import Path
print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA README ===")
print(Path("/data/README.md").read_text())
notes114 = Path("/work/notes.md").read_text()
print("\n=== EARLIER NOTEBOOK ===")
print(notes114[:16000])
print("\n=== KERNEL INVENTORY ===")
print("Functions:", sorted(k for k,v in list(globals().items()) if callable(v) and not k.startswith("_")))
print("Dataframes:", [(k, v.shape) for k,v in list(globals().items()) if isinstance(v, pd.DataFrame)])
print("Forward directories:", {p.name: [q.name for q in p.iterdir()] for p in Path("/work").glob("fwd_*")})
print("Edits:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 114: restart and evidence recovery\n"
            "- Fine-bin conventions do not explain the factual reconstruction gap.\n"
            "- Recovering architecture, early identification evidence, and available forward artifacts before selecting a repair.\n"
            "- No edits. Python runs: 114; forward passes: 4.\n")
```