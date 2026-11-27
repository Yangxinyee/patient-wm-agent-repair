The architecture confirms that drug inputs affect the shared recurrent state, so a broad edit could disturb every prediction head. The preliminary associations are not enough to choose an effect size. I’ll recover the identification checks and repair targets next; agreement across independent estimates would justify a targeted edit.

```python
import pandas as pd
from pathlib import Path

nt = Path("/work/notes.md").read_text()
print("=== IDENTIFICATION NOTEBOOK ===")
print(nt[4300:10200])
print("\n=== SAVED ESTIMATES ===")
for filename in ["kernel_targets.csv", "dynamic_iv.csv", "sensor_iv.csv",
                 "proxy_iv.csv", "local_projections.csv", "dml_results.csv"]:
    df_check = pd.read_csv("/work/" + filename)
    print(filename, df_check.shape, list(df_check.columns))
    print(df_check.head(3).to_string(index=False))
print("\n=== USER-DEFINED FUNCTIONS ===")
print([k for k, v in list(globals().items())
       if callable(v) and getattr(v, "__module__", "") == "__main__"])
print("\n=== LARGE KERNEL OBJECTS ===")
for k, v in list(globals().items()):
    if isinstance(v, (np.ndarray, pd.DataFrame)) and getattr(v, "size", 0) > 1000:
        print(k, type(v).__name__, v.shape)
with open("/work/notes.md", "a") as f:
    f.write("\n## Restart recovery, runs 47–48\n"
            "- Re-read architecture and schema. No active edits. "
            "Action-only posterior reconstruction is exact, but general bundle "
            "reconstruction is not yet trusted. Recovering causal estimates and "
            "cached helpers before choosing a selective repair.\n")
```
