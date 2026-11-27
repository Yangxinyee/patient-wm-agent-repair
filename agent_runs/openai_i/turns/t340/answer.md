Increasing the gains raised factual loss without materially improving the dose effects, supporting the saturation hypothesis. I’ll restore the lower-gain checkpoint and inspect the stronger attention route’s compensation. That route previously produced substantial effects, but it is usable only if its factual overhead can be reduced below the limit.

```python
import inspect, re, numpy as np

with np.load("/work/corrected_direct286.npz", allow_pickle=False) as z291:
    install_checkpoint285({k: z291[k].copy() for k in z291.files})

print("=== RECENT COMPENSATION DESIGN ===")
for tag291 in ["Python 219", "Python 228", "Python 243",
               "Python 272", "Python 274", "Python 276"]:
    for sec291 in sections289:
        if tag291 in sec291.split("\n", 1)[0]:
            print(sec291[:800])
            break

print("=== ROUTING HELPER CONTRACTS ===")
for name291 in ["route_tokens229", "route_features229", "replay_route228",
                "replay_inputs161", "call_core161", "linear136",
                "norm136", "posterior269", "full_route285"]:
    fun291 = globals().get(name291)
    if callable(fun291):
        print(name291, inspect.signature(fun291))
        print("Referenced globals:",
              [x for x in fun291.__code__.co_names if x in globals()])
        print("Local names:", fun291.__code__.co_varnames)

print("=== COMPENSATION / REPLAY VARIABLES ===")
vars291 = [
    k for k in sorted(list(globals()))
    if not k.startswith("_") and not callable(globals()[k])
    and (re.search(r"(275|229|269|243|182|267|268)$", k)
         or re.search(r"empty.*action|action.*empty", k, re.I))
]
for k291 in vars291:
    v291 = globals()[k291]
    if isinstance(v291, dict):
        print(k291, "dict keys:", list(v291.keys())[:16])
    elif hasattr(v291, "shape"):
        print(k291, tuple(v291.shape))
    elif isinstance(v291, (list, tuple)):
        print(k291, type(v291).__name__, "length", len(v291),
              "first", repr(v291[:1])[:160])
    else:
        print(k291, repr(v291)[:160])

print("=== CACHE CONTRACT ===")
if isinstance(args_cache182, dict):
    for k291, v291 in args_cache182.items():
        print(k291, type(v291).__name__, getattr(v291, "shape", None))
else:
    print(type(args_cache182).__name__, "length", len(args_cache182))

with open("/work/notes.md", "a") as f291:
    f291.write(
        "\n## Forwards 35–36; Python 291 — reject higher direct gains\n"
        "- Gains insulin 1 / NE 3 give factual dNLL +.00214, but dose shifts remain "
        "-2.835, +2.120, +.173, +.031.\n"
        "- Direct transport saturates; larger gains offer no meaningful repair.\n"
        "- Restored corrected_direct286 as factual-safe checkpoint.\n"
        "- Investigating corrected-route factual compensation using actual empty-action encoding.\n"
    )
```