The kernel still has head reconstructions, action features, and record-based response estimates. I’ll check their accuracy and dose sensitivity before choosing an edit path. A usable path should distinguish dose changes from the patient condition that prompted treatment; otherwise amplification could reinforce the same confounding.

```python
import inspect, glob

for _name in ("diagnostics27", "decode_table28", "targets29", "irf_table",
              "baseline_readouts23"):
    _v = globals().get(_name)
    print("\n" + _name)
    print(_v.to_string(index=False) if isinstance(_v, pd.DataFrame) else repr(_v))

print("\nHEAD RECONSTRUCTION ACCURACY")
if {"predicted_mean", "local_mean"}.issubset(head_rows.columns):
    _err95 = head_rows["local_mean"] - head_rows["predicted_mean"]
    print(head_rows.assign(reconstruction_error=_err95).groupby("event")
          ["reconstruction_error"].agg(["mean", "std", "min", "max"]).to_string())

print("\nSTAY SOURCES")
_train_ids95 = set(train.stay_id.unique())
_tune_ids95 = set(tuning.stay_id.unique())
for _name in ("head_reference", "repair_train100", "latent_records"):
    _v = globals().get(_name)
    if isinstance(_v, pd.DataFrame):
        _ids95 = set(_v.stay_id.unique())
        print(_name, "stays", len(_ids95),
              "in_train", len(_ids95 & _train_ids95),
              "in_tuning", len(_ids95 & _tune_ids95))

print("\nHELPER CONTRACTS")
for _name in ("head_exact", "latent_action_encode", "latent_hidden",
              "latent_context", "action_vector50", "posterior84", "repair_atom_base"):
    _fn95 = globals().get(_name)
    if callable(_fn95):
        print(_name, inspect.signature(_fn95),
              "dependencies:", _fn95.__code__.co_names,
              "defaults:", _fn95.__defaults__)

print("\nDOSE DIRECTIONS")
for _drug95, _v95 in dose_directions50.items():
    print(_drug95, type(_v95).__name__,
          tuple(_v95.shape) if hasattr(_v95, "shape") else str(_v95)[:700])

print("\nSTATE ARTIFACTS")
for _path95 in sorted(glob.glob("/work/fwd_*/*states*.npz")):
    with np.load(_path95) as _npz95:
        print(_path95, [(k, _npz95[k].shape) for k in _npz95.files])

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Python 95 — pathway selection\n"
             "- Retained head and posterior-action features allow testing a statistical "
             "dose-innovation correction without full recurrent reconstruction.\n"
             "- Checking reconstruction accuracy, feature sensitivity, and training/tuning "
             "provenance before editing; no weights changed.\n")
```