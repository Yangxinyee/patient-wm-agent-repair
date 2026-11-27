The combined carrier and relay barely changed the rollouts, so the relay’s existing output directions are not a useful drug pathway. I’ll restore the relay-only baseline and investigate its two reserved channels. A dedicated channel must measurably change the next-reading prediction in the record-supported direction, not just alter a hidden state.

```python
import re, json

for n in carrier200_names:
    drop_edit(n)
print("Restored relay-only baseline:", edited())
print("Relay validation stays:", json.dumps(relay207_stays))

print("\n=== CPU CORE / HEAD HELPERS ===")
for n, v in sorted(list(globals().items())):
    if callable(v) and (
        n.startswith("rx_") or
        any(k in n.lower() for k in ("core_step", "bundle_step", "value_mean"))
    ):
        if hasattr(v, "__code__"):
            print(n, str(inspect.signature(v)),
                  "globals:", v.__code__.co_names,
                  "constants:", repr(v.__code__.co_consts)[:180])

print("\n=== PRELUDE TENSOR NAMES ===")
print([n for n in NAMES if n.startswith("core.prelude.0")])
print("\n=== CORE CACHE INVENTORY ===")
for n, v in sorted(list(globals().items())):
    if n.startswith(("rx_", "rr_", "rp_")) and not callable(v):
        if hasattr(v, "shape"):
            print(n, tuple(v.shape))
        elif isinstance(v, dict) and any(k in n.lower() for k in ("cache", "state", "data", "weight")):
            print(n, "keys:", list(v)[:15])

print("\n=== EARLIER NUMERIC-EDIT EVIDENCE ===")
text212 = Path("/work/notes.md").read_text()
sections212 = re.split(r"\n(?=## )", text212)
selected212 = [s for s in sections212
               if ("numeric" in s.lower() and
                   any(k in s.lower() for k in ("5x", "nll", "forward", "gain")))]
print("\n".join(s[:950] for s in selected212[:8])[:6500])

with open("/work/notes.md", "a") as f:
    f.write("\n## Forward 20 / Run 212\n"
            "On 25 enriched tuning stays, combined relay+carrier x2 insulin -1.925 "
            "versus original -2.062; half .953 versus .983; NE x2 .227 versus .113; "
            "half -.573 versus -.658. Natural relay output directions remain ineffective. "
            "Restored relay-only baseline; investigating dedicated carrier output channels "
            "and recovering earlier numeric-edit evidence. "
            "Counts: 212 Python runs, 20 forwards.\n")
```