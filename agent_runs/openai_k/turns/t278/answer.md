The retained objects include the rate-feature fits and readout directions, but I still need the assembled correction and its replay interface. I’ll recover those together with the saved rollout targets. This will let me test a concrete weight edit: an improvement must survive multi-hour rollout and the official factual-fit check, rather than merely look promising locally.

```python
import inspect, dis, io, numpy as np, torch, re
r250_items = list(globals().items())

print("=== RECENT OBJECTS, COMPACT ===")
for r250_n, r250_o in r250_items[-180:]:
    if r250_n.startswith(("r246_", "r247_", "r248_", "r249_", "r250_", "__")):
        continue
    if isinstance(r250_o, (np.ndarray, torch.Tensor)):
        print(r250_n, tuple(r250_o.shape))
    elif isinstance(r250_o, dict):
        print(r250_n, "dict", list(r250_o)[:6])
    elif inspect.isfunction(r250_o):
        print(r250_n, "fn", r250_o.__code__.co_varnames[:r250_o.__code__.co_argcount])
    elif isinstance(r250_o, (int, float, str)):
        print(r250_n, repr(r250_o)[:80])
    else:
        print(r250_n, type(r250_o).__name__)

print("\n=== REPLAY SOURCE OR BYTECODE ===")
for r250_n in ["core228", "r179_capture_head"]:
    r250_fn = globals().get(r250_n)
    if r250_fn is None:
        continue
    print("\nFUNCTION", r250_n)
    try:
        print(inspect.getsource(r250_fn)[:4500])
    except (OSError, TypeError):
        for r250_i in dis.get_instructions(r250_fn):
            if r250_i.opname not in ("CACHE", "RESUME", "PRECALL"):
                print(r250_i.offset, r250_i.opname, r250_i.argrepr)

print("\n=== NUMERIC ROLLOUT TARGET NOTES ===")
r250_lines = Path("/work/notes.md").read_text().splitlines()
r250_idxs = [
    i for i, s in enumerate(r250_lines)
    if re.search(r"target|convol", s, re.I) and re.search(r"rollout|tuning|x2|doubl|six.lag", s, re.I)
]
for r250_i in r250_idxs[-12:]:
    print(" ".join(r250_lines[max(0, r250_i-1):r250_i+4])[:500])

with open("/work/notes.md", "a") as r250_file:
    r250_file.write(
        "\n## Run 250: recover assembled recurrent correction\n"
        "- Candidate rate-feature and readout fits are retained; recovering assembled matrix and exact replay interface.\n"
        "- No weights changed. Python runs: 250; forward passes: 17.\n"
    )
```
