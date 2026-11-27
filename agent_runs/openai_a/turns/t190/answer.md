The architecture shows that doses affect predictions through the recurrent state, while the value head also relies on the last recorded reading. That makes a head-only correction vulnerable to feedback errors. I’ll now recover the causal estimates and earlier GPU tests; these should show the required effect sizes and whether the retained edit actually helped.

```python
from pathlib import Path
restart_notes = Path("/work/notes.md").read_text()
print("=== RECORD-BASED IDENTIFICATION AND EARLY TESTS ===")
print(restart_notes[:8500])
print("=== IMPORTANT VARIABLE NAMES ===")
keys = sorted(k for k in globals() if not k.startswith("_"))
print([k for k in keys if any(s in k.lower() for s in
      ["beta", "graft", "base", "causal", "cpu", "safe", "protect", "target", "decode", "candidate"])])
```
