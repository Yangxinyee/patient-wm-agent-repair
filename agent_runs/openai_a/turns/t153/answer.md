The insulin threshold estimates are consistently near −1.9 per additional unit/hour, with no clear pre-treatment discontinuity. Their fitted persistence is about 0.83, implying a much larger sustained effect than the original rollout predicts. I’ll now isolate the remaining reconstruction error; exact state agreement would support fitting a dose decoder from controlled contrasts rather than extrapolating the unstable graft.

```python
def compact_dis(fn, max_chars=2500, min_line=None):
    groups = []
    line = fn.__code__.co_firstlineno
    current = []
    for ins in dis.get_instructions(fn):
        if ins.starts_line is not None:
            if current:
                groups.append(f"{line}: " + " ; ".join(current))
                current = []
            line = ins.starts_line
        if min_line is None or line >= min_line:
            current.append(ins.opname + (" " + ins.argrepr if ins.argrepr else ""))
    if current:
        groups.append(f"{line}: " + " ; ".join(current))
    text = "\n".join(groups)
    print(f"\n=== {fn.__name__} ===\n{text[:max_chars]}")

compact_dis(rec_layer, 2700)
compact_dis(r131_step, 2200, min_line=16)
compact_dis(rec_memory, 2200)

print("\n=== RECURRENT TENSOR SHAPES ===")
for name in NAMES:
    if name.startswith(("core.blocks.0.0", "core.coda", "core.c_out", "core.age")):
        print(name, SHAPES[name])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Recovery after Python 136; forwards 11/50\n"
        "- Insulin fuzzy RD is stable across bandwidths: effect about -1.9/unit-hour; "
        "placebo pre-treatment jumps are near zero.\n"
        "- RD dynamic fit: beta=-1.92870928, persistence=0.83183952. "
        "Earlier pooled sustained x2 target about -11; evaluation-matched stay weighting still needed.\n"
        "- NE sensor instrument is weak; flexible estimates near +3.3 remain assumption-dependent.\n"
        "- Current weights remain the validated beta=-2 head graft. "
        "Investigating recurrent reconstruction before a synthetic-contrast decoder.\n"
    )
```
