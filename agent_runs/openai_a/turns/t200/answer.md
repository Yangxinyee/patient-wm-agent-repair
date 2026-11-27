The recovered notes support an insulin effect near −2 per unit in several adjusted analyses, but the output was truncated before the stronger identification checks and edit machinery. I’ll retrieve those selectively. Agreement across independent checks would justify a stronger edit; conflicting evidence would require revising the target first.

```python
nt = Path("/work/notes.md").read_text()
sections = nt.split("\n## ")
print("=== IDENTIFICATION AND TARGET SECTIONS ===")
budget = 5100
for sec in sections:
    title = sec.splitlines()[0]
    if any(word in title.lower() for word in
           ["discontinu", "threshold", "identification", "causal", "target", "rd ", "rd;", "effect estimate"]):
        excerpt = sec[:min(1800, budget)]
        print("## " + excerpt)
        budget -= len(excerpt)
        if budget < 400:
            break

print("=== KERNEL FUNCTIONS ===")
fnames = [k for k, v in list(globals().items())
          if not k.startswith("_") and callable(v)
          and getattr(v, "__module__", "") == "__main__"]
print(", ".join(fnames))
print("=== KERNEL DICTIONARIES ===")
dnames = [k for k, v in list(globals().items())
          if isinstance(v, dict) and not k.startswith("_")]
print(", ".join(dnames))
print("=== R3 / CHECKPOINT OBJECTS ===")
for k, v in list(globals().items()):
    if k.startswith(("r3", "rollback", "checkpoint", "best", "backup", "base", "causal", "rd_")):
        if isinstance(v, (torch.Tensor, np.ndarray, pd.DataFrame)):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "dict", list(v.keys())[:12])
        elif isinstance(v, (str, float, int, tuple, list)):
            print(k, repr(v)[:140])
```
