The architecture confirms that prediction depends on both the recurrent state and a separate “last reading” register. The output was truncated before the alignment diagnostics, so I still cannot locate the remaining mismatch. I’ll retrieve those objects compactly; agreement in states but not predictions would point to the head or register reconstruction.

```python
print("=== VOCAB ===")
print("keys:", list(VOCAB))
print("config:", VOCAB.get("config"))
print("codes:", VOCAB.get("codes"))
print("=== RECENT OBJECT NAMES ===")
for suffix100 in ("95", "96", "97", "98"):
    names100 = sorted(n for n in globals() if n.endswith(suffix100))
    print(suffix100, names100)
    if suffix100 == "98":
        for n100 in names100:
            o100 = globals()[n100]
            if isinstance(o100, (np.ndarray, torch.Tensor, pd.DataFrame, pd.Series)):
                print(n100, type(o100).__name__, tuple(o100.shape))
            elif isinstance(o100, dict):
                print(n100, "dict", list(o100)[:10])
            elif isinstance(o100, (list, tuple)):
                print(n100, type(o100).__name__, len(o100), repr(o100[:2])[:180])
            else:
                print(n100, repr(o100)[:220])
print("=== IDX80 ===")
print(type(idx80).__name__, getattr(idx80, "shape", None), repr(idx80[:4])[:700])
print("=== DATAFRAME OBJECTS ===")
for n100 in sorted(list(globals())):
    o100 = globals()[n100]
    if isinstance(o100, pd.DataFrame):
        print(n100, o100.shape, list(o100.columns)[:12])
print("=== LOCAL FUNCTION NAMES ===")
print([n for n, o in list(globals().items())
       if inspect.isfunction(o) and not n.startswith("_")])
print("=== IDENTIFICATION NOTES, RUNS 6–10 ===")
start100 = notes99.find("## Run 6:")
end100 = notes99.find("## Run 11", start100)
print(notes99[start100:end100][:3300])
print("=== FORWARD FILES ===")
for d100 in sorted(os.listdir("/work")):
    if d100.startswith("fwd_"):
        print(d100, os.listdir("/work/" + d100))
with open("/work/notes.md", "a") as f100:
    f100.write("\n## Run 100: compact alignment recovery\n"
               "- Architecture uses a recurrent state plus code-specific last-reading registers.\n"
               "- Retrieving recent diagnostic objects and function names without the previous output truncation.\n"
               "- Python runs: 100; forward passes: 4; no submitted edits.\n")
```
