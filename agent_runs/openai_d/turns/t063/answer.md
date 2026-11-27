The edit failed decisively: factual NLL rose from 1.63461 to 38.59464, affecting every reading type. The local gate tests therefore did not predict its actual behavior. I’ll revert all four tensors and inspect the construction before trying another edit; any replacement must first reproduce the real head’s factual outputs.

```python
import re, torch, numpy as np

rejected_head_trial = {
    name: load(name).clone() if isinstance(load(name), torch.Tensor)
    else np.array(load(name), copy=True)
    for name in list(edited())
}
for name in list(edited()):
    drop_edit(name)
print("Edits after rejection:", edited())

print("\n=== RELEVANT PERSISTENT VARIABLES ===")
snapshot = list(globals().items())
pattern = re.compile(r"(head|gate|coef|query|register|latent|state|dml|causal|insulin|norep|panel|train|tune|W0|W2|b0|b2|decoder|record|feature)", re.I)
for name, obj in sorted(snapshot):
    if name.startswith("_") or not pattern.search(name):
        continue
    if isinstance(obj, np.ndarray):
        print(name, "ndarray", obj.shape, obj.dtype)
    elif isinstance(obj, torch.Tensor):
        print(name, "tensor", tuple(obj.shape), obj.dtype)
    elif isinstance(obj, pd.DataFrame):
        print(name, "DataFrame", obj.shape, list(obj.columns)[:12])
    elif isinstance(obj, dict):
        print(name, "dict", list(obj.keys())[:15])
    elif isinstance(obj, (list, tuple)):
        print(name, type(obj).__name__, len(obj), repr(obj)[:220])
    elif isinstance(obj, (str, int, float, np.number)):
        print(name, repr(obj)[:220])
    elif callable(obj):
        print(name, "callable")

print("\n=== HEAD TENSOR SHAPES ===")
for name in NAMES:
    if "attr_heads.value_bin" in name or "last_" in name or "type_emb" in name:
        print(name, SHAPES[name])

print("\n=== NOTEBOOK RUNS 5–10 ===")
text = open("/work/notes.md").read()
start = text.find("## Run 5")
end = text.find("## Run 11")
print(text[start:end][:5200])

try:
    history = get_ipython().history_manager.input_hist_raw
    candidates = [s for s in history if "Installed provisional insulin correction" in s]
    if candidates:
        print("\n=== REJECTED EDIT CONSTRUCTION ===")
        print(candidates[-1][-6500:])
except (NameError, AttributeError):
    pass

with open("/work/notes.md", "a") as f:
    f.write("\n## Forward 7 / Run 55: rejected gated head edit\n"
            "Paired tuning factual NLL: original 1.63461, edited 38.59464. "
            "All four reading codes were damaged, so the local gating/reconstruction "
            "checks were insufficient. Reverted every edited tensor immediately. "
            "Do not retain or merely rescale this unvalidated construction.\n")
```