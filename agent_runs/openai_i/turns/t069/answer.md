The architecture confirms that changing the posterior latent can influence the shared recurrent state, including both sensors and non-value outputs. The recovered record estimates support effects opposite to the naive dose associations. The output was truncated before the helper inventory, so I’ll retrieve that compactly; the next implementation check is whether reconstructed recurrent updates match the saved states.

```python
import inspect, types

lines = ["=== FUNCTIONS ==="]
for name, obj in sorted(list(globals().items())):
    if inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "?"
        lines.append(f"{name}{sig}"[:150])

lines.append("\n=== DATA / STATES / FEATURE OBJECTS ===")
for name, obj in sorted(list(globals().items())):
    if name.startswith("_") or name in {"NAMES", "SHAPES", "VOCAB", "lines"}:
        continue
    if isinstance(obj, pd.DataFrame):
        lines.append(f"{name}: DataFrame {obj.shape}")
    elif isinstance(obj, (np.ndarray, torch.Tensor)) and obj.ndim >= 2:
        lines.append(f"{name}: {type(obj).__name__} {tuple(obj.shape)}")
    elif isinstance(obj, dict):
        keys = list(obj.keys())
        lines.append(f"{name}: dict({len(keys)}) keys={str(keys[:5])[:140]}")
    elif isinstance(obj, (list, tuple)) and len(obj) > 3:
        lines.append(f"{name}: {type(obj).__name__}({len(obj)})")

manifest = "\n".join(lines)
print(manifest[:6500])
print("\nManifest characters:", len(manifest))

print("\n=== CORE AND VALUE TENSOR NAMES ===")
for name in NAMES:
    if name.startswith("core.") or name.startswith("attr_heads.value"):
        print(name, SHAPES[name])

print("\n=== VOCAB STRUCTURE ===")
print("keys:", list(VOCAB.keys()))
print("codes:", VOCAB.get("codes"))
print("Current edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Python 61; forward 6 — compact recovery\n"
        "- Architecture confirms posterior edits feed shared recurrent dynamics, not only "
        "a sensor head. Early crossfit evidence supports insulin near -2 and NE near +3 "
        "per rate unit for the next-hour reading, unlike naive associations.\n"
        "- Previous output hit the display limit; retrieving compact helper/state inventory.\n"
    )
```