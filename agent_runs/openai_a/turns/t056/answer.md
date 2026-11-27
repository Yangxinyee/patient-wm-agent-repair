The new failure came from passing a flattened state array where the helper expects a structured container. It happened before any weight change. I’ll use the original container and rerun the comparison; successful feature extraction will let me test dose decoding next.

```python
import re
from pathlib import Path

print("Head-helper constants:", cpu_head_parts.__code__.co_consts)
for key in (3, 4):
    obj = state_files[key]
    print("State container", key, type(obj).__name__)
    if hasattr(obj, "keys"):
        print([(k, getattr(obj[k], "shape", None)) for k in obj.keys()])

repair_train = repair_head_features(head_meta_train, state_files[4])
repair_tune = repair_head_features(head_meta_tune, state_files[3])
print("Feature shapes:", repair_train["hidden"].shape, repair_tune["hidden"].shape)
for event in ("glucose_finger", "map_arterial"):
    mask = head_meta_tune.event.to_numpy() == event
    print(event,
          "CPU/GPU mean RMSE",
          float(np.sqrt(np.mean((repair_tune["mean"][mask] - head_gpu_mean[mask])**2))),
          "CPU NLL", float(repair_tune["nll"][mask].mean()),
          "GPU NLL", float(head_gpu_nll[mask].mean()))

sections = re.split(r"(?m)^## ", Path("/work/notes.md").read_text())
for section in sections:
    if section.startswith(("Factual baseline and dynamic", "More efficient dynamic")):
        print("\n##", section[:1900])

with open("/work/notes.md", "a") as f:
    f.write("\n## Recovery; Python 50, forward 4\n"
            "- Corrected feature extraction's 17-bin centres and retained structured state input.\n"
            "- Cached approximate original-head features; GPU factual testing remains decisive.\n"
            "- No edits yet; norepinephrine identification is still assumption-sensitive.\n")
print("Edits:", edited())
```
