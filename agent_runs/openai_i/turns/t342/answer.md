The architecture confirms that the routed edit changes shared state, while the value head also uses the last reading. I’ll collect states from the stronger routed candidate and check whether a small, record-fitted calibration can recover its factual loss. This candidate remains provisional: better dose responses alone do not justify submitting it.

```python
import inspect
import pandas as pd
import torch
import dis

print("=== RELEVANT HELPER CONTRACTS ===")
helpers293 = [
    (n, v) for n, v in list(globals().items())
    if callable(v) and hasattr(v, "__code__")
    and re.search(r"(head|readout|predict|view|attr|att_features|core_step|replay).*?\d+$", n)
]
for n, v in helpers293:
    print(n, inspect.signature(v))
    if re.search(r"head|readout|predict|view|attr|att_features", n):
        print("  globals:", v.__code__.co_names)
        print("  locals:", v.__code__.co_varnames)

print("\n=== CALIBRATION DATA CONTRACTS ===")
for n in [
    "state_cal161", "e_cal161", "u_cal161", "stays156", "hours156",
    "indices_all161", "features229", "basefeat275", "basey275",
    "fo275", "hold275", "ix275", "actualx269", "Ubase215",
    "readouts228", "labels182", "targets182", "values182",
    "events182", "centres182", "W", "VOCAB"
]:
    if n in globals():
        v = globals()[n]
        if isinstance(v, dict):
            print(n, "dict keys", list(v)[:12])
        elif hasattr(v, "shape"):
            print(n, tuple(v.shape))
        elif isinstance(v, (list, tuple)):
            print(n, type(v).__name__, "length", len(v))
        else:
            print(n, type(v).__name__)

print("\n=== ATTENTION FEATURE HELPER ===")
if "att_features220" in globals():
    try:
        print(inspect.getsource(att_features220)[:2200])
    except (OSError, TypeError):
        print(dis.Bytecode(att_features220).dis()[:2200])

train_ids293 = []
for chunk293 in pd.read_csv("/data/train.csv", usecols=["stay_id"], chunksize=50000):
    train_ids293 = list(dict.fromkeys(train_ids293 + chunk293.stay_id.tolist()))[:100]
    if len(train_ids293) == 100:
        break
print("\nTRAIN STATE COHORT:", train_ids293)

# Restore the stronger route solely for compensation-data collection.
# Its uncalibrated factual fit is known to be insufficient.
for name293 in list(edited()):
    drop_edit(name293)
for name293, tensor293 in corrected269.items():
    if torch.is_tensor(tensor293):
        tensor293 = tensor293.detach().cpu().numpy()
    save_edit(name293, np.asarray(tensor293, dtype=np.float32))

print("\nInstalled provisional route:", edited())
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Python 293 — routed-candidate calibration data collection\n"
        "- Reinstalled corrected269 provisionally; uncalibrated factual fit is not acceptable.\n"
        "- Preparing a training-stay state cohort and recovering exact readout helpers.\n"
        "- Goal: a small record-fitted value calibration, validated on separate tuning stays, without gradient training of the model.\n"
        "- corrected_direct286 remains the factual-safe restore point.\n"
    )
```