The head-only test failed: safe corrections barely change dose responses, while stronger corrections damage factual fit and still miss norepinephrine. That points to the state update rather than the readout. Next I’ll inspect the dose decoder and posterior-response directions; a viable state correction must respond consistently to dose changes without large factual errors.

```python
import re, os

print("=== POSTERIOR AND LATENT TENSORS ===")
for name in NAMES:
    if any(s in name for s in ("z_post", "z_in", "z_head")):
        print(name, SHAPES[name])

print("\n=== RETAINED DOSE-CORRECTION OBJECTS ===")
for name, value in list(globals().items()):
    if not re.search(r"(39|40|41|42|46|47|49|51|55|56)", name):
        continue
    if isinstance(value, (np.ndarray, torch.Tensor)):
        print(name, tuple(value.shape))
    elif isinstance(value, dict):
        print(name, "dict", len(value), repr(list(value)[:8])[:160])
    elif inspect.isfunction(value):
        print(name, inspect.signature(value))
    elif isinstance(value, (int, float, str)):
        print(name, repr(value)[:100])

print("\n=== POSTERIOR RESPONSE DIAGNOSTICS ===")
print("Target:", target56.tolist(), "scales:", scale56.tolist())
for penalty, direction in D56_candidates.items():
    direction = np.asarray(direction)
    print("penalty", penalty, "direction norms",
          np.linalg.norm(direction, axis=0).tolist())
print("Response mean:", resp56.mean(0).tolist())
print("Response standard deviation:", resp56.std(0).tolist())
for fun in (lin0, mlp0, r51_core):
    print(fun.__name__, "references:", fun.__code__.co_names)

print("\n=== INITIAL NOTES ===")
print(open("/work/notes.md").read()[:3300])
print("\n=== FORWARD FILES ===")
for folder in sorted(os.listdir("/work")):
    if folder.startswith("fwd_"):
        print(folder, os.listdir("/work/" + folder))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 64: reject head-only correction\n"
            "- Rank-one head frontier cannot approach identified drug slopes within factual-fit budget.\n"
            "- Penalty 1 held-out dNLL -0.000555 but insulin x2 slope only -0.371 and NE x2 +0.012.\n"
            "- Stronger corrections damage factual NLL and still miss NE; no candidate submitted.\n"
            "- Investigating dose-centered posterior/state correction using record-based innovations.\n"
            "- Python runs: 64; forward passes: 4; no tensors edited.\n")
```