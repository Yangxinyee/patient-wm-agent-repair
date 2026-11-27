The latest candidate increased local factual NLL by only 0.00350 nats, but its rollout effects remain untested. I’ll recover the architecture and previous validation stays, then check whether the GPU rollouts improve without exceeding the +0.005 factual-loss limit.

```python
from pathlib import Path
import pandas as pd
import re

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\nCURRENT EDITS", edited())

print("\nRELEVANT KERNEL VARIABLES")
for n in sorted(list(globals())):
    if re.search(r"(286|285|280|279|275|266|194)$", n):
        v = globals()[n]
        if isinstance(v, dict):
            print(n, "dict", len(v), "keys", list(v)[:12])
        elif callable(v):
            import inspect
            try:
                print(n, "function", inspect.signature(v))
            except Exception:
                print(n, "callable")
        elif hasattr(v, "shape"):
            print(n, type(v).__name__, v.shape)
        else:
            print(n, type(v).__name__, str(v)[:250])

dirs287 = sorted(
    Path("/work").glob("fwd_*"),
    key=lambda p: int(p.name.split("_")[-1])
)
for d in dirs287[-5:]:
    print("\nFORWARD FILES", d.name, [p.name for p in d.iterdir()])
facts287 = [
    p for d in dirs287 for p in d.glob("factual_edited.csv")
]
if facts287:
    latest_fact287 = facts287[-1]
    df287 = pd.read_csv(latest_fact287)
    print("\nLATEST FACTUAL", str(latest_fact287), df287.shape,
          df287.columns.tolist())
    print(df287.head(2).to_string(index=False))
    idcols287 = [c for c in df287.columns
                 if c in ("stay", "stay_id", "subject_id")]
    if idcols287:
        validation_ids287 = df287[idcols287[0]].drop_duplicates().tolist()
        print("VALIDATION IDS", validation_ids287)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 287\nRecovered architecture, saved-edit metadata, "
            "and latest GPU validation files. No weights changed. "
            "Run-286 candidate still requires GPU rollout and factual checks.\n")
```