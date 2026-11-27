The MAP edit is a sensor-gated linear tilt, so its gain can be changed without modifying the recurrent model. I’ll compare joint gains using two saved MAP fits, including one with stronger factual constraints. A useful candidate should allow a larger MAP correction while keeping held-out value loss comfortably below 0.005.

```python
import time
import numpy as np
import pandas as pd
import torch

t184 = time.time()
mc_train184 = map164_fact
mc_tune184 = mc169
original_head184 = {
    n: torch.as_tensor(np.asarray(load_original(n)), dtype=torch.float32)
    for n in head177_names
}
base_train184 = np.asarray(head177_eval(mc_train184, original_head184)["nll"])
base_tune184 = np.asarray(head177_eval(mc_tune184, original_head184)["nll"])
events_tune184 = np.asarray(cal170_tune_events)

def make_joint184(insulin_gain, map_key, map_gain):
    coef = np.asarray(map164_fits[map_key]) * map_gain
    weights = head177_make(coef)
    ratio = insulin_gain / 1.4
    saved_output = torch.as_tensor(
        np.asarray(best_insulin_v1["attr_heads.value_bin.2.weight"]),
        dtype=torch.float32
    )
    for unit in [170, 89]:
        weights["attr_heads.value_bin.2.weight"][:, unit] += (
            (ratio - 1.0) * saved_output[:, unit]
        )
    return weights

joint_rows184 = []
joint_candidates184 = {}
for map_key184 in [(0.0, 1.0, 0.001), (0.0, 3.0, 0.0001)]:
    for insulin_gain184 in [1.2, 1.3, 1.4]:
        for map_gain184 in [1.5, 2.0, 3.0, 4.0]:
            key184 = (insulin_gain184, map_key184, map_gain184)
            weights184 = make_joint184(*key184)
            joint_candidates184[key184] = weights184
            train_delta184 = (
                np.asarray(head177_eval(mc_train184, weights184)["nll"])
                - base_train184
            )
            tune_delta184 = (
                np.asarray(head177_eval(mc_tune184, weights184)["nll"])
                - base_tune184
            )
            row184 = {
                "insulin_gain": insulin_gain184,
                "map_fw": map_key184[1],
                "map_ridge": map_key184[2],
                "map_gain": map_gain184,
                "train_dNLL": float(train_delta184.mean()),
                "tuning_dNLL": float(tune_delta184.mean())
            }
            for event184 in [
                "glucose_finger", "glucose_lab", "map_arterial", "map_cuff"
            ]:
                row184[event184] = float(
                    tune_delta184[events_tune184 == event184].mean()
                )
            joint_rows184.append(row184)

joint_frontier184 = pd.DataFrame(joint_rows184)
joint_frontier184.to_csv("/work/joint_frontier.csv", index=False)
print(joint_frontier184.to_string(index=False))
print("SECONDS:", round(time.time() - t184, 1))
print("ACTIVE EDITS UNCHANGED:", edited())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 184: joint stochastic factual frontier\n"
        "- Compared insulin gains 1.2/1.3/1.4 and MAP gains 1.5/2/3/4 for two saved MAP regressions.\n"
        "- Evaluated expected four-prior-draw NLL on training and tuning reconstructions.\n"
        "- Saved /work/joint_frontier.csv; no new edits installed.\n"
    )
```