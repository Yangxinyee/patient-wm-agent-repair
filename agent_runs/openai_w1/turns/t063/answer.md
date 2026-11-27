The later checks retain effects near −1.9 glucose units per insulin-rate unit and +3.3 MAP units per norepinephrine-rate unit, without a clear association in the withheld pre-dose assays. That supports—but does not prove—the causal interpretation. I’ll locate the retained candidate fits under their actual names before testing any edit.

```python
print("=== END OF SAVED INVENTORY ===")
print("\n".join(inventory[-145:]))
print("=== OBJECT NAMES CONTAINING RECENT RUN NUMBERS ===")
print([n for n in globals()
       if any(s in n for s in ("48", "49", "50", "51", "trace", "offset"))])
print("=== DATA DOCUMENTATION ===")
print(Path("/data/README.md").read_text()[:2800])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 55\n"
            "Later robustness estimates: insulin one-hour beta -1.9202, NE +3.3339; "
            "withheld-assay diagnostics were compatible with zero. "
            "Recent objects did not use guessed r48_...r51_ prefixes; locating via saved inventory.\n"
            "Counts: 55 Python runs, 6 forward passes. No edits.\n")
```