The architecture confirms that the pilot changes the recurrent latent update, so it could affect both rollouts and factual predictions. The output was truncated, and two forward-result directories may already contain its validation. I’ll inspect those before spending more passes; improved dose shifts with a factual NLL increase below 0.005 would support keeping the pilot.

```python
import re
from pathlib import Path
import numpy as np
import pandas as pd

print("VALIDATION STAYS:", validation_stays)

print("\nPILOT OBJECTS")
for name in sorted(k for k in globals() if re.match(r"r5[0-3]_?", k)):
    obj = globals()[name]
    if hasattr(obj, "shape"):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(name, "dict keys:", list(obj.keys())[:20])
    elif callable(obj):
        try:
            print(name, inspect.signature(obj))
        except Exception:
            print(name, "callable")
    elif isinstance(obj, (int, float, str, list, tuple)):
        print(name, repr(obj)[:250])

print("\nREPAIR FUNCTION NAMES")
print([k for k, v in list(globals().items())
       if inspect.isfunction(v)
       and any(s in k.lower() for s in ("repair", "pilot", "build", "edit"))])

latest_forward_tables = {}
for folder in (Path("/work/fwd_4"), Path("/work/fwd_5")):
    print("\nFORWARD DIRECTORY:", folder)
    for path in sorted(folder.iterdir()):
        print(path.name, path.stat().st_size, "bytes")
        if path.suffix == ".csv":
            table = pd.read_csv(path)
            latest_forward_tables[str(path)] = table
            print("shape:", table.shape, "columns:", table.columns.tolist())
            print(table.head(2).to_string(index=False))
            metric_columns = [
                c for c in table.select_dtypes(include=np.number).columns
                if any(s in c.lower() for s in ("nll", "loss", "shift", "delta", "mean"))
            ]
            print("metric means:", table[metric_columns].mean().to_dict())
            for label_col in ("edit", "intervention", "event", "code"):
                if label_col in table and table[label_col].nunique() <= 20:
                    print("by", label_col)
                    print(table.groupby(label_col)[metric_columns].mean().to_string())
        elif path.suffix in (".json", ".txt"):
            print(path.read_text()[:2000])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 56; forwards 5. Architecture reread. "
            "Inspecting existing forward 4/5 results before requesting more passes; "
            "run 55 output was truncated, not an execution failure.\n")
```