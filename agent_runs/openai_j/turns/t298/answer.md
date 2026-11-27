The hidden features contain some dose sensitivity, but the likelihood reconstruction error—nearly one nat—is too large to trust a new correction’s factual-loss estimate. I’ll check the exact head calculation before changing weights. Agreement with the model’s likelihood would support proceeding; persistent disagreement would rule out this calculation.

```python
import pathlib, inspect, numpy as np

r260_arch = pathlib.Path("/model/ARCHITECTURE.md").read_text()
print("=== ARCHITECTURE ===")
print(r260_arch)

print("\n=== SAVED GEOMETRY ===")
with np.load("/work/hidden_geometry259.npz", allow_pickle=False) as r260_z:
    for r260_k in r260_z.files:
        r260_a = r260_z[r260_k]
        print(r260_k, r260_a.shape, r260_a.dtype,
              r260_a.tolist() if r260_a.size <= 16 else "")

print("\n=== RELEVANT HELPERS: names and argument names only ===")
r260_globals = list(globals().items())
for r260_k, r260_v in r260_globals:
    if inspect.isfunction(r260_v) and any(
        r260_s in r260_k.lower()
        for r260_s in ["head", "nll", "hidden", "moment", "feature",
                       "install", "candidate", "mixture", "calib"]
    ):
        r260_c = r260_v.__code__
        print(r260_k, r260_c.co_varnames[:r260_c.co_argcount],
              "source:", r260_c.co_filename)

print("\n=== RECENT KERNEL OBJECTS ===")
r260_recent = [(k, v) for k, v in r260_globals
               if not k.startswith(("__", "r260_"))][-65:]
for r260_k, r260_v in r260_recent:
    if hasattr(r260_v, "shape"):
        print(r260_k, type(r260_v).__name__, tuple(r260_v.shape))
    elif isinstance(r260_v, dict):
        print(r260_k, "dict keys:", list(r260_v)[:18])
    elif isinstance(r260_v, (list, tuple)):
        print(r260_k, type(r260_v).__name__, "length", len(r260_v),
              repr(r260_v[:4])[:200])
    elif isinstance(r260_v, (int, float, str)):
        print(r260_k, repr(r260_v)[:200])

print("\nCURRENT EDITS:", edited())
with open("/work/notes.md", "a") as r260_f:
    r260_f.write(
        "\n## Run 260: likelihood audit before further edits\n"
        "- Run 259 likelihood reconstruction differed by up to 0.9695 nats; "
        "do not trust its factual-loss approximation until reconciled.\n"
        "- Reading architecture and recovering saved hidden geometry/helpers. "
        "No weights changed.\n"
    )
```