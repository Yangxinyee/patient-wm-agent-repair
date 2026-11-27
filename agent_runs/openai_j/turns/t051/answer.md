The original weights are restored. The model has a separate action encoder and latent update, which may offer a more selective repair than scaling all numerical inputs. I’ll inspect those paths and the cached helpers. The candidate approach is to transmit dose deviations separately from patient condition; held-out decoding and dose sensitivity must support that separation before editing.

```python
import inspect

print("=== LATENT / STATE TENSORS ===")
for name in NAMES:
    if any(term in name for term in (
        "z_prior", "z_post", "z_in", "z_head", "latent",
        "query_norm", "type_emb", "core.gap", "core.out_norm"
    )):
        print(name, SHAPES[name])

print("\n=== RECENT CACHED ARRAYS / TABLES ===")
cached = [
    (key, value) for key, value in list(globals().items())
    if not key.startswith("_") and isinstance(value, (np.ndarray, pd.DataFrame))
]
for key, value in cached[-55:]:
    print(key, type(value).__name__, value.shape)

print("\n=== USER-DEFINED HELPERS ===")
for key, value in list(globals().items()):
    if inspect.isfunction(value) and getattr(value, "__module__", "") == "__main__":
        try:
            signature = str(inspect.signature(value))
        except Exception:
            signature = "?"
        print(key, signature)

print("\n=== NUMERIC VOCAB EXCERPT ===")
print(str(VOCAB["numeric"])[:2200])

print("\n=== STATE FILE SCHEMA ===")
for directory in sorted(Path("/work").glob("fwd_*")):
    for path in directory.glob("states_*.npz"):
        with np.load(path) as archive:
            print(str(path), {key: archive[key].shape for key in archive.files})

with notes_path.open("a") as f:
    f.write(
        "\n- Run 43: inspecting latent/action pathways and cached helpers for "
        "an innovation-based algebraic repair. No edits. "
        "Any candidate needs held-out factual and intervention checks.\n"
    )
```