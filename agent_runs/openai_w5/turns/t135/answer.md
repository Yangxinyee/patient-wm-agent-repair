The architecture confirms that dose information must reach the recurrent state to affect rollouts. Earlier record estimates suggest substantial, persistent drug effects, while the original model’s dose responses are nearly zero. I’ll recover the recent encoder and routing helpers next; accurate rate encoding alone does not establish that this repair will work.

```python
import re

print("=== EARLIER EFFECT CHECKS ===")
print(notes_text[3500:6500])

print("\n=== LOCAL FUNCTION NAMES ===")
fn_names = [
    n for n, o in list(globals().items())
    if callable(o) and getattr(o, "__module__", "") == "__main__"
    and not n.startswith("_")
]
print(", ".join(sorted(fn_names)))

print("\n=== RECENT CANDIDATE OBJECTS ===")
pattern = re.compile(
    r"(10[0-9]|11[0-9]|channel|private|analytic|reserve|ratefit|rate_fit|"
    r"encoder|prelude|causal|dynamic|effect|tune_ids|tuning_ids|baseline)"
)
lines = []
for n, o in sorted(list(globals().items())):
    if n.startswith("_") or not pattern.search(n):
        continue
    if isinstance(o, (np.ndarray, torch.Tensor)):
        lines.append(f"{n}: {type(o).__name__} {tuple(o.shape)}")
    elif isinstance(o, dict):
        lines.append(f"{n}: dict keys={list(o.keys())[:12]}")
    elif isinstance(o, (list, tuple, int, float, str)):
        lines.append(f"{n}: {repr(o)[:160]}")
print("\n".join(lines)[:4300])

print("\n=== ALL GLOBAL NAMES ===")
print(", ".join(sorted(
    n for n in globals()
    if not n.startswith("_") and n not in {"notes_text", "fn_names", "lines"}
))[:1800])
```
