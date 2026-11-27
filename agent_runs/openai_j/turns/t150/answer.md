The tuning states are now available, with no weights changed. I’ll align them to the factual readings and inspect the register and retained candidate tensors. This will let us distinguish a wrong input reconstruction from an edit that simply extrapolates badly.

```python
import torch
import dis

tune_states = np.load("/work/fwd_14/states_original.npz")
tune_factual_original = pd.read_csv("/work/fwd_13/factual_original.csv")
tune_factual_rejected = pd.read_csv("/work/fwd_13/factual_edited.csv")

print("=== VOCAB STRUCTURE ===")
print("keys:", list(VOCAB))
for key in VOCAB:
    if any(s in key.lower() for s in ("norm", "config", "code")):
        value = VOCAB[key]
        print(key, str(value)[:1500])

print("\n=== HEAD COMPUTATION TENSORS ===")
for name in NAMES:
    if any(s in name for s in (
        "gap", "z_prior", "z_head", "query_norm", "out_norm",
        "last_numeric", "last_status", "type_emb"
    )):
        print(name, SHAPES[name])

print("\n=== HELPER DEPENDENCIES ===")
for name in ("reconstruct_reading_head", "corrected_register_head"):
    fn = globals()[name]
    print(name, "names:", fn.__code__.co_names)
    print("constants:", str(fn.__code__.co_consts)[:1300])
print("\ncorrected_register_head instructions:")
instructions = list(dis.Bytecode(corrected_register_head))
print("\n".join(str(x) for x in instructions)[:3500])

print("\n=== RETAINED HEAD-SIZED ARRAYS ===")
for name, obj in list(globals().items()):
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        if tuple(obj.shape) in ((192,), (192, 192), (17,), (17, 192)):
            print(name, tuple(obj.shape))
    elif isinstance(obj, dict) and any(
        s in name.lower() for s in ("repair", "candidate", "weight", "gate")
    ):
        print(name, "dict keys:", list(obj)[:15])

print("\n=== TUNING STATE TIMES, FIRST STAY ===")
sel = tune_states["stay_id"] == 2020000000
print(tune_states["hour"][sel][:12])
print(tuning[tuning.stay_id == 2020000000].head(12).to_string(index=False))
```
