Numeric gain 4 is restored. The architecture has separate continuous-number and fine-bin inputs for drugs, so either could limit dose sensitivity. The output was truncated before the earlier tests and saved helpers; I’ll recover those compactly to avoid repeating failed edits.

```python
import re

print("=== SAVED FUNCTIONS ===")
for r112_n, r112_o in sorted(list(globals().items())):
    if callable(r112_o) and getattr(r112_o, "__module__", "") == "__main__":
        print(r112_n)

print("\n=== SAVED ARRAYS / DATAFRAMES ===")
r112_items = []
for r112_n, r112_o in sorted(list(globals().items())):
    if r112_n.startswith(("r111_", "r112_", "_")):
        continue
    if isinstance(r112_o, (np.ndarray, pd.DataFrame)):
        r112_items.append((r112_n, type(r112_o).__name__, r112_o.shape))
print(r112_items)

print("\n=== EARLIER EDIT RESULTS ===")
r112_early = r111_notes.split("## Restart / Python runs 49")[0]
for r112_l in r112_early.splitlines():
    if (r112_l.startswith("##") or
        re.search(r"gain|scale|bin|shear|ablat|target|persist|rollout|NLL|action", r112_l, re.I)):
        print(r112_l[:240])

print("\n=== ACTION / EMBEDDING SHAPES ===")
for r112_n in NAMES:
    if r112_n.startswith(("embed.", "encode_action.", "core.z_post", "core.z_in")):
        print(r112_n, SHAPES[r112_n])

print("\n=== VOCAB KEYS ===", list(VOCAB))
print("\n=== CURRENT OPS ===", ops())
```