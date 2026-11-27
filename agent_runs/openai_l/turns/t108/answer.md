The model has separate action and posterior pathways, so a more selective repair is possible in principle. I’ll now check the estimated time course and the validated one-step interfaces. A useful correction must respond to altered doses while remaining small on recorded treatment histories; otherwise it will repeat the head-only trade-off.

```python
import inspect, numpy as np
from pathlib import Path

print("=== RECORD-ESTIMATED RESPONSE KERNELS ===")
for n100 in ("kernels19", "kernel_se19", "kernels22"):
    o100 = globals().get(n100)
    if isinstance(o100, dict):
        for k100, v100 in o100.items():
            try:
                a100 = np.asarray(v100)
                print(n100, k100, a100.shape, np.round(a100, 4))
            except Exception:
                print(n100, k100, str(v100)[:500])

print("\n=== VALIDATED COMPUTATION DEPENDENCIES ===")
for n100 in ("w63", "wc37", "posterior50", "bundle_inputs74",
             "context64", "transition64", "make_cf81",
             "head_inputs96", "latent_probe93"):
    f100 = globals().get(n100)
    if not callable(f100):
        continue
    print("\n", n100, inspect.signature(f100))
    try:
        s100 = inspect.getsource(f100)
        print(s100[:1800])
    except Exception:
        c100 = f100.__code__
        print("locals:", c100.co_varnames)
        print("globals:", c100.co_names)
        print("constants:", tuple(x for x in c100.co_consts
                                  if isinstance(x, (str, int, float, tuple)))[:35])

print("\n=== COUNTERFACTUAL INPUT CONTENTS ===")
for n100 in ("c88", "cf_inputs96"):
    o100 = globals().get(n100)
    if isinstance(o100, dict):
        if n100 == "cf_inputs96":
            o100 = o100.get(("insulin", 2.0), {})
        if isinstance(o100, dict):
            for k100, v100 in o100.items():
                print(n100, k100, type(v100).__name__, getattr(v100, "shape", None))
                if hasattr(v100, "columns"):
                    print("columns:", list(v100.columns))
                    print(v100.head(2).to_string(index=False))

print("\n=== CORE RETAINED ARRAY NAMES ===")
snap100 = dict(globals())
for n100, o100 in sorted(snap100.items()):
    if n100.endswith(("49", "60", "65", "74", "81", "82", "94")):
        sh100 = getattr(o100, "shape", None)
        if sh100 is not None and not callable(o100):
            print(n100, tuple(sh100))

with Path("/work/notes.md").open("a") as f100:
    f100.write(
        "\n## Run 100 / causal timing and selective-path audit\n"
        "Inspecting retained causal response kernels and validated one-step dependencies. "
        "Next candidate must encode dose deviations from observational policy, rather than "
        "apply a broad value-head shift. No weights edited.\n"
    )
```
