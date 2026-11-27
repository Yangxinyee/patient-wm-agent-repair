The ablation passed the factual check (+0.00249 nats), but its drug responses were weaker than numeric gain 4 alone. I’ll remove it and review the earlier targeted edits. The next hypothesis is that a selective drug-signal correction could outperform uniform amplification; stronger rollouts with little factual cost would support that.

```python
import inspect

undo()
print("Restored operations:", ops())

notes = Path("/work/notes.md").read_text()
print("\n=== EARLY EXPERIMENTS, COMPACT ===")
start = notes.find("## Run 4")
end = notes.find("## Python run 35")
early = notes[start:end if end > start else start + 14000]
for line in early.splitlines():
    if len(line) > 700:
        line = line[:700] + " ... [long line shortened]"
    print(line)

print("\n=== FUNCTIONS ===")
for k, v in list(globals().items()):
    if callable(v) and getattr(v, "__module__", "") == "__main__":
        try:
            print(k, inspect.signature(v))
        except Exception:
            print(k)

print("\n=== ARRAYS / TABLES / DICTIONARIES ===")
for k, v in list(globals().items()):
    if k.startswith("_") or k in {"v", "notes", "early"}:
        continue
    if isinstance(v, np.ndarray):
        print(k, v.shape)
    elif isinstance(v, pd.DataFrame):
        print(k, "table", v.shape)
    elif isinstance(v, dict) and k not in {"VOCAB", "SHAPES", "__builtins__"}:
        print(k, "dict", len(v), list(v)[:5])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 68: prior-head ablation rejected\n"
            "- Factual increase +0.00249 nats passed the probe constraint, but all drug responses remain weak.\n"
            "- Removed z_head.weight ablation; restored numeric gain 4 alone.\n"
            "- Reviewing earlier selective edits and available state estimators. Forward passes: 27.\n")
```
