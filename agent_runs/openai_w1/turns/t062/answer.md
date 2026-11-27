The records suggest insulin lowers later glucose and norepinephrine raises later MAP, with effects lasting several hours. Those estimates remain conditional on adequate adjustment for dose assignment. I’ll retrieve the remaining robustness checks and candidate matrices; consistent placebo checks and transferable dose responses would support proceeding.

```python
print("=== CAUSAL ROBUSTNESS AND REPAIR PLAN ===")
print(notes_text[10500:14900])
print("=== RECENT OBJECTS ONLY ===")
recent_names = [n for n in globals()
                if n.startswith(("r48_", "r49_", "r50_", "r51_"))]
for n in recent_names:
    o = globals()[n]
    if isinstance(o, (np.ndarray, torch.Tensor)):
        description = f"{type(o).__name__}{tuple(o.shape)}"
    elif isinstance(o, dict):
        description = f"dict[{len(o)}] keys={list(o)[:8]}"
    elif isinstance(o, (tuple, list)):
        description = f"{type(o).__name__}[{len(o)}]"
    elif isinstance(o, (str, int, float, bool)):
        description = repr(o)[:100]
    else:
        description = type(o).__name__
    print(n, description)
print("=== COUNTS ===")
print("recent_objects", len(recent_names), "current_edits", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 54\n"
            "Recovered multi-hour dose-response estimates. "
            "Retrieving remaining confounding checks and recent candidate matrices without broad inventories.\n"
            "Counts: 54 Python runs, 6 forward passes. No edits.\n")
```