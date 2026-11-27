The latest factual loss increased by 0.00412 nats per reading, within the 0.005 limit on this cohort. That supports keeping the blend provisionally, but does not establish its dose accuracy. I’ll recover the other audits and the records-based effect estimates before choosing final rollout tests. Consistent factual fit and better dose shifts would support retaining it; a budget breach or worse shifts would favor rollback.

```python
import os, re, json
import numpy as np
import pandas as pd
import torch

_snapshot299 = dict(globals())

def _array299(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

print("ARCHITECTURE")
print(open("/model/ARCHITECTURE.md").read())
print("\nRECORD SCHEMA")
print(open("/data/README.md").read())

_notes299 = open("/work/notes.md").read()
print("\nRECORD-BASED IDENTIFICATION EXCERPTS")
_sections299 = re.split(r"(?m)(?=^## )", _notes299)
_evidence299 = [
    s for s in _sections299
    if re.search(r"DML|threshold|kernel23|Run 23\b|identification", s, re.I)
]
print("Matching section headings:")
for s in _evidence299:
    print(s.splitlines()[0][:180])
_substantive299 = [
    s for s in _evidence299
    if not re.search(r"^## (?:Run[s]? )?(?:27[5-9]|28\d|29\d)\b", s)
]
for s in _substantive299[-12:]:
    print("\n" + s[:2200])

print("\nRECENT FORWARD ARTIFACTS")
_audits299 = {}
for k in range(30, 44):
    folder = f"/work/fwd_{k}"
    if not os.path.isdir(folder):
        continue
    entry = {}
    for model in ("original", "edited"):
        path = f"{folder}/factual_{model}.csv"
        if not os.path.isfile(path):
            continue
        d = pd.read_csv(path)
        nll_cols = [c for c in d.columns if "nll" in c.lower()]
        if nll_cols:
            preferred = next(
                (c for c in ("nll", "value_nll", "fine_nll", "bin_nll")
                 if c in nll_cols), nll_cols[0]
            )
            entry[model] = (len(d), float(pd.to_numeric(d[preferred]).mean()))
        else:
            entry[model] = {"columns": d.columns.tolist()}
    if entry:
        if isinstance(entry.get("original"), tuple) and isinstance(entry.get("edited"), tuple):
            entry["delta"] = entry["edited"][1] - entry["original"][1]
        _audits299[k] = entry
        print(folder, entry)
    if k >= 38:
        for model in ("original", "edited"):
            path = f"{folder}/dose_edit_{model}.csv"
            if os.path.isfile(path):
                d = pd.read_csv(path)
                print(path, "shape", d.shape, "columns", d.columns.tolist())
                print(d.head(8).to_string(index=False))
                group_cols = [c for c in d.columns
                              if c.lower() in ("edit", "intervention", "dose_edit", "scenario")]
                shift_cols = [c for c in d.columns if "shift" in c.lower()]
                if group_cols and shift_cols:
                    print(d.groupby(group_cols[0])[shift_cols].mean().to_string())

print("\nCURRENT EDITS AND BACKUPS")
print(edited())
for name in ("backup296", "backup292_strong", "backup297_current", "grid286"):
    obj = _snapshot299.get(name)
    print(name, type(obj).__name__,
          list(obj.keys())[:12] if isinstance(obj, dict) else "not a dictionary")
if isinstance(_snapshot299.get("backup296"), dict) and isinstance(_snapshot299.get("backup292_strong"), dict):
    for name in ("attr_heads.value_bin.2.weight", "attr_heads.value_bin.2.bias"):
        try:
            intended = (0.8 * _array299(_snapshot299["backup296"][name])
                        + 0.2 * _array299(_snapshot299["backup292_strong"][name]))
            print(name, "blend maximum error",
                  float(np.max(np.abs(_array299(load(name)) - intended))))
        except Exception as e:
            print(name, "verification unavailable:", repr(e))

_tuning299 = pd.read_csv("/data/tuning.csv")
_idcol299 = next(
    (c for c in ("stay_id", "stay", "id") if c in _tuning299.columns), None
)
print("\nTUNING COLUMNS", _tuning299.columns.tolist())
if _idcol299 is not None:
    _ids299 = _tuning299[_idcol299].drop_duplicates().tolist()
    _cohorts299 = {j: _ids299[100*j:100*(j+1)] for j in range(4)}
    for j, ids in _cohorts299.items():
        print("COHORT", j, json.dumps(ids))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 299 / final audit recovery\n"
        "No weights changed. Current edit is the final-head-only 80/20 blend of "
        "backup296 and backup292_strong; other five edited tensors are unchanged. "
        "Forward41 (fwd_43), tuning cohort300–399, reported factual delta +0.00412 "
        "over4676 readings. Recovered architecture, record schema, identification "
        "notes, recent artifact scores, backup structure and tuning cohort IDs. "
        "One Python run remains, reserved for final selection and notebook completion.\n"
        + "Recovered factual artifact scores: " + repr(_audits299) + "\n"
    )
```
