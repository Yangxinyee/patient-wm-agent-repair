The architecture confirms that doses enter both the posterior and the prelude, so an upstream edit could affect every glucose or MAP sensor rather than just one output head. The notebook also shows why the early state-space estimates were rejected. I’ll recover the accepted response estimates and exact interfaces next; a useful edit must reproduce those responses, not merely reverse the model’s current dose sensitivity.

```python
import inspect
import re

print("=== FUNCTIONS ===")
funcs106 = {
    k: v for k, v in list(globals().items())
    if inspect.isfunction(v) and getattr(v, "__module__", "") == "__main__"
}
for k106, v106 in funcs106.items():
    print(k106, str(inspect.signature(v106)))

print("\n=== RECENT VARIABLE INTERFACES ===")
for k106, v106 in list(globals().items()):
    if k106.endswith(("101", "102", "103", "104")):
        if hasattr(v106, "shape"):
            print(k106, tuple(v106.shape))
        elif isinstance(v106, dict):
            print(k106, "keys", list(v106)[:12])

print("\n=== POSTERIOR TENSORS ===")
for n106 in NAMES:
    if any(s106 in n106 for s106 in
           ("z_post", "z_prior", "z_in", "z_head", "prelude_norm")):
        print(n106, SHAPES[n106])

print("\n=== ACCEPTED EVIDENCE / RESPONSE NOTES ===")
sections106 = notes105.split("\n## ")
chosen106 = [
    s for s in sections106[12:]
    if any(t in s.lower() for t in
           ("-1.94", "3.05", "response kernel", "causal persistence",
            "accepted target", "corrected dml", "distributed lag"))
]
budget106 = 2600
for s106 in chosen106:
    fragment106 = s106[:min(650, budget106)]
    print("##", fragment106)
    budget106 -= len(fragment106)
    if budget106 <= 0:
        break

print("\nCurrent edits:", edited())
```
