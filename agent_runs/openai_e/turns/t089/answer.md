Arterial and cuff analyses both support persistence near 0.70, and both pre-treatment placebos are compatible with zero. Their x2 targets are about +4.0 to +4.6 MAP units. I’ll now recover the MAP-directed edit machinery and preserve the insulin baseline. A useful repair must move both norepinephrine readouts toward these targets without violating factual fit.

```python
baseline_b77 = as_numpy67(load("core.b_in.weight")).copy()
baseline_original77 = as_numpy67(load_original("core.b_in.weight")).copy()
baseline_delta77 = baseline_b77 - baseline_original77
np.savez("/work/baseline77.npz", delta=baseline_delta77)
print("Baseline delta norm:", np.linalg.norm(baseline_delta77))
print("Difference from 1.6*candidate58:",
      np.linalg.norm(baseline_delta77 - 1.6*candidate_delta58))

text77 = Path("/work/notes.md").read_text()
heads77 = list(re.finditer(r"^## .*", text77, re.M))
sections77 = []
for k, match in enumerate(heads77):
    if re.search(r"Run (4[5-8]|5[3-7])\b", match.group()):
        end = heads77[k+1].start() if k+1 < len(heads77) else len(text77)
        sections77.append(text77[match.start():end])
print("\nEDIT DEVELOPMENT NOTES\n" + "\n".join(sections77)[:4700])

lines77 = []
for name, obj in list(globals().items()):
    if name.startswith("_") or inspect.isfunction(obj):
        continue
    if not (re.search(r"(46|47|52|56|57)$", name) or re.search(r"(direction|initial|null|admission|precision|anchor)", name, re.I)):
        continue
    if isinstance(obj, dict):
        lines77.append(f"{name}: keys={list(obj)[:12]}")
    elif hasattr(obj, "shape"):
        lines77.append(f"{name}: {type(obj).__name__}{tuple(obj.shape)}")
    elif isinstance(obj, (int, float)):
        lines77.append(f"{name}: {obj}")
print("\nEDIT ARRAYS\n" + "\n".join(lines77)[:3100])

for name in ["coda46", "next_means45", "prelude46", "feature_metrics57"]:
    fun77 = globals().get(name)
    if fun77 is not None:
        print(name, "names:", fun77.__code__.co_names,
              "constants:", repr(fun77.__code__.co_consts)[:160])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 77: freeze insulin baseline and recover NE edit machinery\n"
            "Independent arterial/cuff persistence is about .70; placebo effects are near zero. "
            "Working NE x2 target about +4.2 to +4.6, half about -2.1 to -2.3. "
            "Saved current insulin-only b_in delta to /work/baseline77.npz before NE surgery. "
            "No weights changed. Python runs: 77; forwards: 10.\n")
```
