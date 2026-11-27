The architecture confirms that dose edits act through recurrent state and a shared reading head, so stronger edits can change feedback unexpectedly. The early estimates support an insulin effect near −2 per unit-hour, while the norepinephrine estimates change sign with adjustment. I’ll recover the identification tests and graft helpers next; consistent discontinuity estimates would support a causal target, but ordinary adjusted associations alone would not.

```python
import re
note_sections = re.split(r"\n(?=## )", notes_text)
print("=== NOTEBOOK INDEX ===")
for i, s in enumerate(note_sections):
    print(i, s.splitlines()[0][:110])
print("=== IDENTIFICATION SECTIONS ===")
selected = [(i, s) for i, s in enumerate(note_sections)
            if i < 30 and any(x in s.lower() for x in
                              ("discontinu", "fuzzy", "threshold", "placebo"))]
budget = 3800
for i, s in selected:
    chunk = f"\nSECTION {i}\n{s}\n"
    if budget <= 0:
        break
    print(chunk[:budget])
    budget -= len(chunk)
print("=== RELEVANT RETAINED OBJECTS ===")
for k in sorted(list(globals())):
    if k.startswith("_") or not any(s in k.lower() for s in
        ("graft", "gate", "ridge", "rd_", "panel", "step", "checkpoint",
         "tuning_x", "train_x", "dose", "transition3")):
        continue
    obj = globals()[k]
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        desc = f"{type(obj).__name__} {tuple(obj.shape)}"
    elif isinstance(obj, dict):
        desc = f"dict {len(obj)} keys={list(obj)[:5]}"
    elif callable(obj):
        desc = "function " + str(getattr(getattr(obj, "__code__", None),
                                          "co_varnames", ())[:7])
    else:
        desc = type(obj).__name__
    print(k, desc)
```
