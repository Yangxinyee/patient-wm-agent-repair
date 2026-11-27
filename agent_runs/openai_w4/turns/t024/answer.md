```python
import os, json, inspect
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())

print("\n=== RECORD SCHEMA ===")
print(open("/data/README.md").read())

print("\n=== EARLIER NOTEBOOK CONTEXT ===")
_notes = open("/work/notes.md").read()
_cut = _notes.find("## Run 7")
print(_notes[:_cut if _cut >= 0 else 6500][-8500:])

print("\n=== RETAINED OBJECTS ===")
for _name, _obj in list(globals().items()):
    if _name.startswith("_") or _name in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(_obj, pd.DataFrame):
        print(_name, "DataFrame", _obj.shape, "columns:", list(_obj.columns)[:12])
    elif isinstance(_obj, np.ndarray) and _obj.size > 20:
        print(_name, "array", _obj.shape, str(_obj.dtype))
    elif isinstance(_obj, dict):
        print(_name, "dict keys:", list(_obj.keys())[:18])
    elif inspect.isfunction(_obj) and getattr(_obj, "__module__", "") == "__main__":
        try:
            print(_name, "function", str(inspect.signature(_obj)))
        except Exception:
            print(_name, "function")

print("\n=== EXISTING FORWARD SUMMARIES ===")
for _folder in sorted(os.listdir("/work")):
    if not _folder.startswith("fwd_"):
        continue
    _path = "/work/" + _folder
    print(_folder, os.listdir(_path))
    for _file in os.listdir(_path):
        if _file.startswith("dose_edit_") and _file.endswith(".csv"):
            _df = pd.read_csv(_path + "/" + _file)
            _valid = _df[_df["reached"].astype(str).str.lower().isin(["true", "1"])]
            print(_file, "eligible rows:", len(_valid))
            print(_valid.groupby("edit")["shift"].agg(["mean", "std", "count"]).round(5).to_string())
        elif _file.endswith(".json"):
            _txt = open(_path + "/" + _file).read()
            print(_file, _txt[:2500])

print("\nCurrent edits:", edited())
with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 20 (restart)\n"
             "Recovered architecture, record schema, earlier notebook context, retained objects, "
             "and original rollout summaries. The provisional one-exponential impulse model is "
             "not accepted as truth; dose-innovation identification and flexible lag responses "
             "need checking before a weight repair. No weights edited.\n")
```