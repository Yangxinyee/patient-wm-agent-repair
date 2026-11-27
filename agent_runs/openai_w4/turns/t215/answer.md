The recovered architecture shows how the dose signal enters the recurrent state. I’ll now separate gate suppression from raw decoding error. If the gate is already saturated on active bundles, strengthening it will not fix the attenuation; the decoder itself will need recalibration.

```python
import inspect
import dis
import io
from scipy.special import expit

def show_source190(name, limit=4500):
    obj = globals().get(name)
    if obj is None:
        return
    print("\nFUNCTION", name, inspect.signature(obj))
    try:
        text = inspect.getsource(obj)
    except (OSError, TypeError):
        buf = io.StringIO()
        dis.dis(obj, file=buf)
        text = buf.getvalue()
    print(text[:limit])

show_source190("gated_ne185", 3000)
show_source190("collect_selected187", 4500)
show_source190("capture_transition102", 4500)

print("\nSELECTED RAW COEFFICIENT")
matches190 = []
for key, value in raw_candidates87.items():
    a = np.asarray(value)
    if a.shape == coef_ne186.shape:
        matches190.append((float(np.linalg.norm(a - coef_ne186)), key))
print(sorted(matches190, key=lambda x: x[0])[:5])

print("\nGATE MARGINS ON ALL FIVE AUGMENTATIONS")
n190 = len(ix180)
for j in range(len(raws180)):
    sl = slice(j * n190, (j + 1) * n190)
    scores = gate_fits180[100.0].decision_function(xx180[sl])
    active = np.asarray(active_ne180, dtype=bool)
    held = np.asarray(test180, dtype=bool)
    for label, mask in (
        ("held active", held & active),
        ("held inactive", held & ~active),
    ):
        s = scores[mask]
        print(j, label, "n", len(s),
              "margin quantiles", np.round(np.quantile(s, [0, .01, .1, .5, .9, 1]), 5),
              "mean sigmoid40", float(expit(40 * s).mean()))
    print("raw augmentation", j, type(raws180[j]).__name__,
          getattr(raws180[j], "shape", None))

print("\nRELEVANT RAW/TARGET ARRAYS")
for name, value in list(globals().items()):
    if (name.endswith(("87", "102", "185", "187"))
        and any(word in name.lower() for word in
                ("raw", "target", "innov", "coef", "rate", "hold", "train", "test"))):
        print(name, summary189(value))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 190\n"
        "Inspecting exact gate/channel interfaces and active gate margins over "
        "all retained dose augmentations. This distinguishes gate suppression "
        "from raw decoder approximation/extrapolation error before any edit. "
        "No saved weights changed.\n"
    )
```