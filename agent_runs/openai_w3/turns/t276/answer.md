The notebook records strong, roughly linear insulin effects and positive norepinephrine effects after history adjustment; the original model largely missed them. The previous output was truncated before the implementation details, so it did not test the saturation hypothesis. I’ll retrieve only those details next; a dominant transport component would give us a reversible, interpretable experiment.

```python
import types
lines246 = ["EDIT SCALES"]
for n246 in edited():
    a246 = np.asarray(load(n246))
    o246 = np.asarray(load_original(n246))
    lines246.append(f"{n246}: original={np.linalg.norm(o246):.3g}, delta={np.linalg.norm(a246-o246):.3g}")
lines246.append(f"dd193 norms {np.linalg.norm(dd193, axis=0)}")

lines246.append("\nFUNCTIONS")
for k246, v246 in sorted(list(globals().items())):
    if isinstance(v246, types.FunctionType) and v246.__module__ == "__main__":
        lines246.append(f"{k246}{v246.__code__.co_varnames[:v246.__code__.co_argcount]}")

lines246.append("\nTRANSPORT OBJECTS")
for k246, v246 in sorted(list(globals().items())):
    if any(s246 in k246 for s246 in ("193", "206", "220", "185")):
        if isinstance(v246, (np.ndarray, torch.Tensor)) and (v246.ndim <= 1 or 192 in tuple(v246.shape)):
            lines246.append(f"{k246}: {type(v246).__name__} {tuple(v246.shape)}")
        elif isinstance(v246, (int, float)) and not k246.startswith(("i", "j", "k")):
            lines246.append(f"{k246}: {v246}")

lines246.append("\nHELPER REFERENCES")
for k246 in ("prelude155", "preludeE185", "prelude_parts145", "prelude_cache88"):
    f246 = globals()[k246]
    lines246.append(f"{k246}: names={f246.__code__.co_names}; defaults={f246.__defaults__}")
out246 = "\n".join(lines246)
print(out246[:8800])
print("Inventory characters:", len(out246))
with open("/work/notes.md", "a") as f246:
    f246.write("\nPython run 246; forwards 18. Bounded transport/helper inventory after output truncation. "
               "Record identification supports approximately linear insulin dose effects and positive NE effects. "
               "Checkpoint222 remains installed; no new edits.\n")
```