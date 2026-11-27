The 1,250-second candidate passes the factual limit on the first 100 stays: NLL rises by only 0.00058 nats. Its insulin responses are still below the provisional record-derived targets. I’ll reversibly increase only the glucose readout strength by 40%; this is useful only if both insulin responses increase without violating factual fit.

```python
from pathlib import Path
from safetensors.torch import save_file
import numpy as np
import torch

print("Sensor neuron pairs:", sensor_neurons)
glucose_columns289 = []
for sensor in ["glucose_finger", "glucose_lab"]:
    pair = sensor_neurons[sensor]
    assert isinstance(pair, (list, tuple)) and len(pair) == 2, (sensor, pair)
    assert all(isinstance(i, (int, np.integer)) for i in pair), (sensor, pair)
    glucose_columns289.extend(int(i) for i in pair)
assert len(set(glucose_columns289)) == 4

BASE289 = {
    name: torch.as_tensor(load(name)).float().clone().contiguous()
    for name in edited()
}
save_file(BASE289, "/work/checkpoint_run289.safetensors")

head_weight289 = BASE289["attr_heads.value_bin.2.weight"].clone()
head_weight289[:, glucose_columns289] *= 1.40
save_edit("attr_heads.value_bin.2.weight", head_weight289)

print("Scaled glucose columns:", glucose_columns289)
print("Insulin persistence and factual centers unchanged; NE strength unchanged.")
print("Current edited tensors:", len(edited()))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 289: reversible glucose-strength trial\n"
        "- Tau=1250 baseline factual A: 1.639911 vs 1.639331, "
        "delta +0.000580; passes locally.\n"
        "- Saved all pre-trial edits in checkpoint_run289.safetensors.\n"
        "- Multiplied only the four glucose signal/control output columns "
        "by 1.40. Transport, persistence, centers, and NE unchanged.\n"
        "- Candidate is unaccepted pending paired factual and rollout checks, "
        "including fresh tuning stays. No gradient training.\n"
    )
```