The current edit passes the factual-loss limit on all 400 tuning stays, but norepinephrine remains under-responsive. The differing insulin targets need reconciliation before any further weight change. I will check the exact response kernels and averaging conventions; agreement on identical rows would support the calibration, while disagreement would argue against increasing its strength.

```python
import os, re, json
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
audit297_architecture = open("/model/ARCHITECTURE.md").read()
print(audit297_architecture[:11000])

audit297_notes = open("/work/notes.md").read()
print("\n=== IDENTIFICATION AND TARGET CONVENTIONS ===")
audit297_sections = re.split(r"(?=^## )", audit297_notes, flags=re.M)
for audit297_section in audit297_sections:
    if re.match(r"^## Run (6|131|132)\b", audit297_section):
        print(audit297_section[:2800])

def audit297_compact(x):
    if isinstance(x, np.ndarray):
        return x.tolist() if x.ndim <= 1 and x.size <= 16 else {
            "shape": x.shape, "dtype": str(x.dtype)
        }
    if isinstance(x, pd.DataFrame):
        return x.to_string(index=False) if len(x) <= 8 else {
            "shape": x.shape, "columns": list(x.columns)
        }
    if isinstance(x, pd.Series):
        return x.to_dict() if len(x) <= 16 else {"length": len(x)}
    if isinstance(x, dict):
        return {str(k): audit297_compact(v) for k, v in list(x.items())[:12]}
    if isinstance(x, (list, tuple)):
        return [audit297_compact(v) for v in x] if len(x) <= 12 else {
            "length": len(x)
        }
    if callable(x) and hasattr(x, "__code__"):
        return {"references": x.__code__.co_names}
    if hasattr(x, "shape"):
        return {"shape": tuple(x.shape)}
    if isinstance(x, (str, int, float, bool, np.number, type(None))):
        return x
    return type(x).__name__

print("\n=== EXACT RETAINED CONVOLUTION INPUTS ===")
audit297_namespace = dict(globals())
for audit297_name in sorted(audit297_namespace):
    if (
        audit297_name.startswith("r295")
        or re.search(r"^r132_.*(?:ir|target|slope|rho)$", audit297_name)
        or re.search(r"(impuls|kernel)", audit297_name, re.I)
    ):
        audit297_value = audit297_namespace[audit297_name]
        print(audit297_name, str(audit297_compact(audit297_value))[:1500])

print("\n=== FINAL POOLED ROLLOUT SUMMARY ===")
for audit297_filename in (
    "final_rollout400_summary.csv",
    "final_rollout400_record_comparison.csv",
):
    audit297_path = "/work/" + audit297_filename
    print(audit297_filename)
    print(pd.read_csv(audit297_path).to_string(index=False))

print("\n=== EXACT SUBMITTED-WEIGHT AUDIT ===")
print("Edited tensors:", edited())
audit297_current = load("core.b_in.weight")
if hasattr(audit297_current, "detach"):
    audit297_current = audit297_current.detach().cpu().numpy()
audit297_current = np.asarray(audit297_current)
with np.load("/work/margin_validated400.npz", allow_pickle=False) as audit297_archive:
    print("Validated archive keys:", audit297_archive.files)
    for audit297_key in audit297_archive.files:
        audit297_array = audit297_archive[audit297_key]
        if audit297_array.shape == audit297_current.shape:
            print(
                audit297_key,
                "max absolute difference from current:",
                float(np.max(np.abs(audit297_current - audit297_array))),
            )

print("\n=== FORWARD SUMMARY METADATA ===")
for audit297_folder in ("fwd_37", "fwd_38", "fwd_40", "fwd_41", "fwd_42"):
    audit297_directory = "/work/" + audit297_folder
    print(audit297_folder, os.listdir(audit297_directory))
    for audit297_filename in os.listdir(audit297_directory):
        if audit297_filename.endswith(".json"):
            with open(os.path.join(audit297_directory, audit297_filename)) as audit297_file:
                audit297_metadata = json.load(audit297_file)
            print(audit297_filename, str(audit297_compact(audit297_metadata))[:1800])

with open("/work/notes.md", "a") as audit297_file:
    audit297_file.write(
        "\n## Run 297: final calibration and submission audit\n"
        "- Recovered architecture, retained convolution inputs, pooled rollout summaries, "
        "exact submitted-weight archive comparison, and available forward metadata.\n"
        "- No weights changed. Current candidate retains its previously measured "
        "400-stay factual delta of +0.003384797 nats per reading.\n"
        "- Record-derived targets remain estimates requiring identification assumptions, "
        "not observed intervention truth.\n"
        "- Python runs: 297; forward passes: 44.\n"
    )
```