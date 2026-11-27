The audit found the failed trial and a previously GPU-checked backup, but has not yet explained the mismatch. I’ll restore the checked combined edit and inspect the trial’s construction. Changes outside the intended neurons would indicate an installation error; otherwise I’ll investigate the local reconstruction.

```python
import re
import inspect

def c180_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

def c180_brief(x, depth=0):
    if hasattr(x, "shape"):
        return f"{type(x).__name__}{tuple(x.shape)}"
    if isinstance(x, dict):
        if depth >= 2:
            return f"dict keys={list(x)[:12]}"
        return {k: c180_brief(v, depth + 1) for k, v in list(x.items())[:18]}
    if isinstance(x, (list, tuple)):
        return [c180_brief(v, depth + 1) for v in x[:8]]
    return x if isinstance(x, (str, int, float, bool, np.number)) else type(x).__name__

print("=== INSTALLED VS BACKUPS ===")
for name in r171_combined_backup:
    current = c180_np(load(name))
    print(name)
    for label, backup in [
        ("validated_combined", r171_combined_backup),
        ("run178_fallback", r178_fallback),
        ("run178_candidate", r178_candidate),
    ]:
        print(" ", label, "max difference",
              float(np.max(np.abs(current - c180_np(backup[name])))))

for name, tensor in r171_combined_backup.items():
    assert tuple(c180_np(tensor).shape) == tuple(SHAPES[name])
    assert np.isfinite(c180_np(tensor)).all()
    save_edit(name, tensor)
print("Restored the GPU-checked combined candidate.")

print("=== RECENT CONSTRUCTION RESULTS ===")
for name in ["r175_screen", "r177_results", "r178_rows"]:
    print(name, c180_brief(globals()[name]))

print("=== CACHE STRUCTURES ===")
for name in ["cache", "cache64", "cache81", "cache98",
             "head_records", "head_meta", "candidate_inputs",
             "r177_draws", "r178_rows"]:
    if name in globals():
        print(name, c180_brief(globals()[name]))

print("=== AVAILABLE LOCAL FUNCTIONS ===")
c180_functions = [
    (name, obj) for name, obj in list(globals().items())
    if inspect.isfunction(obj) and not name.startswith("_")
]
for name, obj in c180_functions:
    if any(s in name.lower() for s in
           ["head", "logit", "loss", "gate", "ne", "probe", "candidate", "nll", "decode"]):
        print(name, str(inspect.signature(obj)))

c180_notes = Path("/work/notes.md").read_text()
print("=== EARLY RECORD-BASED FINDINGS ===")
print(c180_notes[:5000])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 180: restore checked combined fallback\n"
            "Restored r171_combined_backup, whose GPU factual delta was "
            "+0.00385475 on tuning 300..399 and rollout shifts were "
            "insulin -14.331/+6.691, NE +2.387/-0.873 on tuning 0..99. "
            "The failed run-178 trial is not accepted. Auditing its "
            "construction before any further trial.\n")
```