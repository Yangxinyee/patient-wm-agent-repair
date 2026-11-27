The retained helper already uses the interleaved numeric layout, but we still have not verified that it matches the model’s actual inputs. I’ll stop the verbose code inspection and measure where the prediction errors occur, alongside the original dose-rollout results. Errors concentrated in register-bearing rows would point toward register construction rather than state reconstruction.

```python
import numpy as np
import pandas as pd
from pathlib import Path

print("=== BASELINE FORWARD RESULTS ===")
for directory in sorted(Path("/work").glob("fwd_*")):
    print(directory.name, [p.name for p in directory.iterdir()])
    for path in directory.glob("dose_edit_*.csv"):
        table = pd.read_csv(path)
        print(path.name, "shape:", table.shape, "columns:", list(table.columns))
        print(table.head(6).to_string(index=False))
        if "edit" in table.columns:
            print(table.groupby("edit").mean(numeric_only=True).to_string())

print("\n=== RETAINED DATA OBJECTS ===")
for name, obj in list(globals().items()):
    if hasattr(obj, "shape") and not callable(obj) and name not in ("train", "tuning", "df", "W"):
        print(name, type(obj).__name__, obj.shape)
    elif isinstance(obj, dict) and name not in (
        "VOCAB", "SHAPES", "head_weights", "__builtins__"
    ):
        print(name, "keys:", list(obj)[:8])

print("\n=== STATE AND RECORD ALIGNMENT ===")
print("state_hours:", np.asarray(state_hours)[:35])
print(train[train.stay_id == head_rows.stay_id.iloc[0]].head(35).to_string(index=False))
for path in Path("/work").glob("fwd_*/states_original.npz"):
    with np.load(path) as archive:
        print(path, {k: archive[k].shape for k in archive.files})

print("\n=== RECONSTRUCTION ERROR BREAKDOWN ===")
result20 = head2(head_rows, z_samples=32)
print("return type:", type(result20).__name__)
if isinstance(result20, tuple):
    print("return shapes:", [getattr(x, "shape", None) for x in result20])
    mean20 = np.asarray(result20[0]).reshape(-1)
else:
    mean20 = np.asarray(result20).reshape(-1)
if len(mean20) == len(head_rows):
    check20 = head_rows[["event", "hour", "last_value", "predicted_mean", "nll"]].copy()
    check20["has_last"] = check20.last_value.notna()
    check20["error"] = mean20 - check20.predicted_mean.to_numpy()
    check20["squared_error"] = check20.error ** 2
    print(check20.groupby(["event", "has_last"]).agg(
        n=("error", "size"), bias=("error", "mean"),
        mse=("squared_error", "mean")).to_string())
    print(check20.assign(hour_band=pd.cut(check20.hour, [-3, 0, 6, 12, 25]))
          .groupby(["event", "hour_band"], observed=True)
          .agg(n=("error", "size"), bias=("error", "mean"),
               mse=("squared_error", "mean")).to_string())

with open("/work/notes.md", "a") as f:
    f.write("\n## Behavioral reconstruction checks — Python 20; forward 4\n"
            "- Inspecting original rollout results and reconstruction errors by sensor, register presence and hour.\n"
            "- No weights changed.\n")
```