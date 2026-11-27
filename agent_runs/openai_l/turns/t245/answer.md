The actual factual increase is +0.01412, so this candidate fails the constraint despite its favorable offline validation. I’ll roll it back and inspect the feature construction and available caches. I’ll test stronger regularization next; the hypothesis would be supported if it preserves dose sensitivity while closing the offline-to-GPU loss gap.

```python
import inspect, numpy as np
from pathlib import Path

for n220 in list(edited()):
    if n220 not in backup219:
        drop_edit(n220)
for n220, v220 in backup219.items():
    save_edit(n220, torch.from_numpy(v220.copy()))
print("Restored accepted conservative edit:", edited())

print("\n=== DOSE DIRECTION STRUCTURE ===")
print(brief218(directions207))
print("\n=== HELPER FUNCTIONS ===")
for n220, v220 in sorted(list(globals().items())):
    if callable(v220) and getattr(v220, "__module__", "") == "__main__":
        try:
            print(n220, inspect.signature(v220))
        except Exception:
            print(n220, "(signature unavailable)")

print("\n=== STATE / DATA CACHE OBJECTS ===")
for n220, v220 in sorted(list(globals().items())):
    if not callable(v220) and re.search(
        r"(state|cache|batch|panel|train|tuning|stay|raw204|offset204|family204|Lorig179|C179)", n220, re.I):
        if isinstance(v220, (np.ndarray, torch.Tensor)):
            print(n220, tuple(v220.shape))
        elif isinstance(v220, dict):
            print(n220, "dict keys", list(v220)[:10])
        elif isinstance(v220, pd.DataFrame):
            print(n220, v220.shape, v220.columns.tolist()[:16])
        elif isinstance(v220, (list, tuple)):
            print(n220, type(v220).__name__, len(v220), repr(v220[:3])[:120])

with Path("/work/notes.md").open("a") as f220:
    f220.write("\n## Forward 13 / nonlinear eta=1 rejected\n"
               "100-stay tuning GPU factual NLL 1.64576 vs 1.63164 (+.01412), "
               "despite offline validation +.00178. Rejected and rolled back "
               "to backup219 (five-tensor conservative edit, factual +.00263). "
               "Need diagnose dose-direction generalization / insufficient "
               "distillation coverage; no causal rollout tested for this candidate.\n")
```
