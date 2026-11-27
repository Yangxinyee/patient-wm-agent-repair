The insulin response increases smoothly with connection strength, with little MAP leakage. Scale 1 roughly matches the record-based immediate coefficient; larger scales may improve the sustained rollout, but that must be measured directly. I’ll provisionally save scale 2, which left factual-fit margin in training replay, and test the evaluation rollout. A substantially larger, stable glucose shift would support it.

```python
import re

backup_before167 = {n: torch.as_tensor(load(n)).clone() for n in edited()}
candidate167 = dict(channel_weights160)
candidate167["core.b_in.weight"] = (
    channel_weights160["core.b_in.weight"] + 2.0 * outer166
)
save_edit("core.b_in.weight", candidate167["core.b_in.weight"])
print("Saved ridge-0.1 connection at scale 2.0.")
print("Current edited tensors:", edited())

print("\nLONGITUDINAL ESTIMATE STRUCTURE")
for drug167, fit167 in lp_new136.items():
    print(drug167, {
        str(k): (tuple(v.shape) if hasattr(v, "shape")
                 else list(v.keys())[:12] if isinstance(v, dict)
                 else v if np.isscalar(v)
                 else type(v).__name__)
        for k, v in fit167.items()
    })

print("\nCHANNEL-DESIGN NOTEBOOK EXCERPTS")
notes167 = Path("/work/notes.md").read_text()
for run167 in [58, 73, 90, 95, 100, 103, 110, 120, 123, 137]:
    match167 = re.search(
        rf"(?m)^## Run {run167}\b.*?(?=^## |\Z)", notes167, flags=re.S | re.M)
    if match167:
        print(match167.group(0)[:430].strip())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 167\nSaved provisional ridge-0.1 B_in response at scale 2.0 "
            "(channel_weights160 plus 2*outer166); previous saved edits backed up in "
            "backup_before167. Training replay dNLL for this direction/scale was "
            "+0.001832. Run-166 impulse response is immediate and reading-mediated; "
            "actual free-rollout validation is next. No inference of truth from "
            "model impulse responses.\n")
```