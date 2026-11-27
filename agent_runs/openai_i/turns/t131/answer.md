The records suggest insulin lowers glucose and norepinephrine raises MAP after controlling for treatment selection, while the original rollouts show much weaker responses. Those one-hour estimates are not yet the cumulative targets. I’ll recover the lag estimates and encoder helpers, then test dose decoding before choosing an edit.

```python
import inspect, re
from pathlib import Path

print("=== RELEVANT NOTEBOOK SECTIONS ===")
audit_text = Path("/work/notes.md").read_text()
audit_sections = re.split(r"(?=^## )", audit_text, flags=re.M)
audit_selected = []
for audit_section in audit_sections:
    audit_heading = audit_section.splitlines()[0] if audit_section else ""
    if any(term in audit_section.lower() for term in
           ("impulse", "convolution", "persistence", "lag target", "cumulative", "rd estimate")):
        audit_selected.append(audit_section[:1000])
print("\n".join(audit_selected[:9])[:5800])

print("\n=== FUNCTIONS (NAMES ONLY) ===")
audit_items = list(globals().items())
print(" ".join(n for n, o in audit_items
               if inspect.isfunction(o) and getattr(o, "__module__", "") == "__main__"
               and not n.startswith("audit_")))

print("\n=== POSTERIOR AND ACTION TENSORS ===")
for audit_name in NAMES:
    if any(s in audit_name for s in
           ("core.z_post", "core.z_in", "encode_action", "embed.numeric")):
        print(audit_name, SHAPES[audit_name])

print("\n=== RECENT RETAINED OBJECTS ===")
audit_relevant = []
for audit_name, audit_obj in audit_items:
    if audit_name.startswith("audit_"):
        continue
    if hasattr(audit_obj, "shape"):
        audit_relevant.append((audit_name, type(audit_obj).__name__, audit_obj.shape))
    elif isinstance(audit_obj, dict) and any(
        s in audit_name.lower() for s in ("post", "dose", "drug", "decod", "reg", "hidden", "latent", "causal")):
        audit_relevant.append((audit_name, "dict", list(audit_obj)[:8]))
for audit_entry in audit_relevant[-50:]:
    print(*audit_entry)

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 121 — recover cumulative targets and earlier dose pathway\n"
            "- Architecture confirms dose information enters a separate action encoder before the shared posterior/state.\n"
            "- Testing earlier dose representations rather than trusting failed final-hidden decoders.\n"
            "- No weights changed.\n")
```