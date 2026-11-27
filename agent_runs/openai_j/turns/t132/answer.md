The notebook supports immediate drug effects, but insulin persistence is still uncertain. The output was truncated before the target estimates and cached variables appeared. I’ll recover those compactly; stable fixed-dose estimates would justify a rollout target, while unstable estimates would call for a conservative edit.

```python
import re, os

audit_notes = Path("/work/notes.md").read_text()
print("=== NOTEBOOK SECTIONS ===")
print("\n".join(re.findall(r"^##.*$", audit_notes, flags=re.M))[:2200])

audit_start = audit_notes.find("## Fixed-dose response-kernel")
print("\n=== FIXED-DOSE EVIDENCE AND BASELINE ===")
print(audit_notes[audit_start:audit_start + 4100])

print("\n=== TARGET TABLE ===")
audit_targets = pd.read_csv("/work/kernel_targets.csv")
print(audit_targets.to_string(index=False)[:1600])

print("\n=== CACHED ARRAYS / TABLES ===")
audit_vars = []
audit_functions = []
for audit_name, audit_obj in list(globals().items()):
    if audit_name.startswith("_") or audit_name.startswith("audit_"):
        continue
    if isinstance(audit_obj, (pd.DataFrame, pd.Series, np.ndarray)):
        audit_vars.append(f"{audit_name}:{tuple(audit_obj.shape)}")
    elif callable(audit_obj) and getattr(audit_obj, "__module__", "") == "__main__":
        audit_functions.append(audit_name)
    elif isinstance(audit_obj, dict) and audit_name not in ("SHAPES",):
        audit_vars.append(f"{audit_name}:dict({len(audit_obj)})")
print("  ".join(audit_vars)[:3300])
print("Functions:", ", ".join(audit_functions)[:1500])
print("\nForward 12 files:", os.listdir("/work/fwd_12"))
```
