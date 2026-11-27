The architecture confirms that dose magnitude enters the recurrent state through numeric and bin embeddings. Removing that magnitude while retaining drug identity, timing, and presence is therefore a plausible edit—but only if factual validation supports it.

The output was truncated before the earlier edit results. I’ll retrieve those results and the available analysis objects next. A small NLL cost and dose-invariant rollouts would support keeping the edit; a larger cost would require revising it.

```python
notes_text = Path("/work/notes.md").read_text()
start = notes_text.find("## Run 10")
stop = notes_text.find("## Python run 21", start)
if stop < 0:
    stop = notes_text.find("## Run 21", start)
if stop < 0:
    stop = start + 6500
print("=== EARLIER EDIT AND VALIDATION RESULTS ===")
print(notes_text[start:stop][:7000])

print("\n=== AVAILABLE OBJECTS ===")
objects = list(globals().items())
print("Names:", ", ".join(
    k for k, v in objects
    if not k.startswith("_") and k not in {"NAMES", "SHAPES", "VOCAB"}
))
print("\nDataframes and arrays:")
for k, v in objects:
    if k.startswith("_"):
        continue
    if type(v).__name__ in {"DataFrame", "Series", "ndarray", "Tensor"}:
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict) and any(
        term in k.lower() for term in ["result", "report", "effect", "fit", "lag"]
    ):
        print(k, "keys:", list(v)[:15])

print("\n=== FORWARD OUTPUT FILES ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in folder.iterdir()])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Python runs 24–25: restart audit\n"
        "- Reread architecture and record definitions; no weight changes.\n"
        "- Retrieving earlier edit-validation results and persistent analysis objects.\n"
        "- Current evidence: insulin effects remain small and uncertain; conditional lag contrasts are not rollout estimates.\n"
        "- Python runs: 25. Forward passes: 7.\n"
    )
```
