Zeroing absent actions made the reconstruction worse, so I reject that hypothesis. The retained unit maps also disagree. I’ll use isolated readings and exposed states to select the convention empirically, without assuming medical units. Near-numerical agreement across sensors would validate it; larger errors would show that another input convention remains wrong.

```python
import numpy as np
import pandas as pd
import torch

valid_set78 = set(np.asarray(valid_indices65, dtype=int).tolist())
isolated78 = []
for event78 in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff"):
    found78 = 0
    for row78 in latent_records.loc[latent_records.event == event78].to_dict("records"):
        key78 = (int(row78["stay_id"]), int(round(float(row78["hour"]) * 60)))
        index78 = bundle_lookup65.get(key78)
        if index78 is None or int(index78) not in valid_set78:
            continue
        index78 = int(index78)
        if float(e_counts65[index78]) != 1 or float(u_counts65[index78]) != 0:
            continue
        isolated78.append((event78, index78, row78))
        found78 += 1
        if found78 == 10:
            break

print("=== ISOLATED READING COUNTS ===")
print(pd.Series([x[0] for x in isolated78]).value_counts().to_string())
print("=== EXAMPLE ASSEMBLED ROW ===")
if isolated78:
    print(repr(bundle_rows65[isolated78[0][1]])[:700])

indices78 = np.array([x[1] for x in isolated78], dtype=int)
ctx78 = latent_context(indices78)
truth78 = torch.from_numpy(latent_state[indices78, -32:])
empty_u78 = empty_set58("encode_action").reshape(1, 192)

unit_scores78 = []
for unit78 in range(6):
    atoms78 = torch.stack([
        complete_atom57(row78, kind=0, ekind=1, unit=unit78)
        for _, _, row78 in isolated78
    ])
    encoded78 = single_set57(atoms78, "encode")
    for absent78 in ("encoded", "zero"):
        action78 = (empty_u78.expand(len(indices78), -1)
                    if absent78 == "encoded"
                    else torch.zeros(len(indices78), 192))
        predicted78 = latent_post_mean(ctx78, encoded78, action78)
        errors78 = (predicted78 - truth78).square().mean(1).sqrt().numpy()
        for event78 in sorted(set(x[0] for x in isolated78)):
            mask78 = np.array([x[0] == event78 for x in isolated78])
            unit_scores78.append({
                "event": event78, "unit": unit78, "absent_action": absent78,
                "mean_rms": float(errors78[mask78].mean()),
                "max_rms": float(errors78[mask78].max())
            })

unit_table78 = pd.DataFrame(unit_scores78).sort_values(["event", "mean_rms"])
print("\n=== BEST THREE CONVENTIONS PER SENSOR ===")
print(unit_table78.groupby("event", sort=False).head(3).to_string(index=False))
print("\n=== VOCABULARY TOP-LEVEL KEYS ===", list(VOCAB))
print("=== COMPLETE-ATOM DEFAULTS ===", complete_atom57.__defaults__)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Python 78; forward 6 — empirical categorical validation\n"
        "- Setting absent actions to zero worsened posterior agreement; hypothesis rejected.\n"
        "- Retained unit mappings disagree. Tested unit IDs and absent-action "
        "conventions on isolated readings against exposed states, without medical assumptions.\n"
        "- No weights changed.\n"
    )
```