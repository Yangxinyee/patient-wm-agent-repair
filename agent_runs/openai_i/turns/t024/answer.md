The tensor inventory shows that numeric drug inputs are separate from reading inputs, which may allow a targeted edit later. First, I need to resolve the register layout; the last output was cut off before that code appeared. I’ll print only the relevant sections and the baseline dose results. A sharp reconstruction improvement would support the missing-field hypothesis.

```python
import dis
import pandas as pd
from pathlib import Path

print("=== COMPACT FUNCTION DETAILS ===")
for name in ["reconstruct_head", "head2", "_fn"]:
    fn = globals().get(name)
    if not callable(fn) or not hasattr(fn, "__code__"):
        continue
    print("\n", name, "defaults:", fn.__defaults__)
    print("variables:", fn.__code__.co_varnames)
    print("constants:", fn.__code__.co_consts)
    instructions = list(dis.get_instructions(fn))
    keep = set()
    for i, instruction in enumerate(instructions):
        if instruction.argval in (
            "last_z", "last_numeric", "last_status.weight",
            "last_bin", "last_hour", "interleaved", "register"
        ):
            keep.update(range(max(0, i - 8), min(len(instructions), i + 18)))
    for i in sorted(keep):
        ins = instructions[i]
        print(ins.offset, ins.opname, ins.argrepr)

print("\n=== RETAINED SOURCE STRINGS ===")
for name, obj in list(globals().items()):
    if isinstance(obj, str) and ("def reconstruct_head" in obj or "def head2" in obj):
        print(name, obj)

print("\n=== RETAINED DATA OBJECTS ===")
for name, obj in list(globals().items()):
    if hasattr(obj, "shape") and not callable(obj) and name not in ("train", "tuning", "df"):
        print(name, type(obj).__name__, obj.shape)
    elif isinstance(obj, dict) and name not in (
        "VOCAB", "SHAPES", "head_weights", "__builtins__"
    ):
        print(name, "keys:", list(obj)[:12])

print("\n=== FORWARD FILES AND BASELINE DOSE RESULTS ===")
for directory in sorted(Path("/work").glob("fwd_*")):
    print(directory.name, [p.name for p in directory.iterdir()])
    for path in directory.glob("dose_edit_*.csv"):
        table = pd.read_csv(path)
        print(path.name, table.shape, list(table.columns))
        print(table.head(8).to_string(index=False))
        numeric_cols = table.select_dtypes("number").columns
        if "edit" in table.columns:
            print(table.groupby("edit")[numeric_cols].mean().to_string())

with open("/work/notes.md", "a") as f:
    f.write("\n## Compact helper inspection — Python 19; forward 4\n"
            "- Previous bytecode output was truncated; isolating register construction and baseline dose results.\n"
            "- No weights changed.\n")
```