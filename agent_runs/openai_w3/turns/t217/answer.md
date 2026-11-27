The architecture confirms that dose signals pass through several nonlinear stages, so simply scaling the current edit could worsen the asymmetry. I’ll inspect the retained calibration helpers and paired dose contrasts next. Accurate decoding in both directions would point to the downstream transport—not the dose decoder—as the main problem.

```python
import inspect, dis

print("=== KEY HELPER SIGNATURES ===")
for name in ["posterior153", "latent187", "hidden192", "preludeE185",
             "response186", "phys171", "view171", "prelude_parts145"]:
    f = globals().get(name)
    if callable(f):
        print(name, inspect.signature(f), "dependencies:", f.__code__.co_names)
        try:
            src = inspect.getsource(f)
            print(src[:1800])
        except (OSError, TypeError):
            pass

print("\n=== REMAINING RECENT OBJECTS ===")
for name in sorted(g194):
    if re.search(r"(186|187|192|193)", name) and name[0].lower() >= "s":
        v = g194[name]
        if hasattr(v, "shape"):
            print(name, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(name, "dict keys", list(v)[:15])
        elif isinstance(v, (tuple, list)):
            print(name, type(v).__name__, len(v))
        elif isinstance(v, (int, float, bool)):
            print(name, v)

print("\n=== PAIRED LATENT DATA ===")
for key in [(0, 0.5), (0, 2.0), (1, 0.5), (1, 2.0)]:
    v = pairs187[key]
    print(key, [(type(x).__name__, tuple(x.shape) if hasattr(x, "shape") else repr(x)[:80])
                for x in v])

print("\n=== PILOT CONSTRUCTION OBJECTS ===")
for name in ["post154", "zin154", "zb154", "pilot192", "best192", "p192",
             "decoder192", "directions187", "Tmain187", "mu187", "gate187"]:
    v = globals().get(name)
    if isinstance(v, dict):
        print(name, {k: tuple(x.shape) if hasattr(x, "shape") else type(x).__name__
                     for k, x in v.items()})
    elif hasattr(v, "shape"):
        a = np.asarray(v)
        print(name, a.shape, "norm", float(np.linalg.norm(a)),
              "range", (float(a.min()), float(a.max())))
    else:
        print(name, repr(v)[:300])

print("\n=== INSTALLED PILOT DELTAS ===")
for name in checkpoint193:
    a = np.asarray(checkpoint193[name])
    b = np.asarray(base192[name])
    print(name, "delta norm", round(float(np.linalg.norm(a-b)), 6))

print("\n=== RECORD-IDENTIFICATION NOTE EXCERPT ===")
print(Path("/work/notes.md").read_text()[:3500])

print("\n=== FORWARD OUTPUT FILES ===")
for k in [11, 12]:
    print(k, [p.name for p in Path(f"/work/fwd_{k}").iterdir()])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 195; forwards 12. Architecture recovered. Inspecting paired normalized-latent contrasts and exact calibration helpers to distinguish bidirectional dose-decoding error from nonlinear transport. No weights changed.\n")
```