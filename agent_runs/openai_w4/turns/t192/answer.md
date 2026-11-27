The latest factual check worsened NLL by only **0.00018 nats**, below the 0.005 limit on these 100 stays. That supports factual preservation, but not the dose effects. I’ll recover the matching stay IDs and identification context before testing free rollouts. Responses moving toward the record-based estimates would support the edit; unchanged or unstable responses would refute its usefulness.

```python
import json, pathlib, numpy as np, pandas as pd

print("ARCHITECTURE")
print(pathlib.Path("/model/ARCHITECTURE.md").read_text())

factual168 = pd.read_csv("/work/fwd_13/factual_original.csv")
print("\nFACTUAL OUTPUT COLUMNS:", factual168.columns.tolist())
staycol168 = next(c for c in factual168.columns if "stay" in c.lower())
rollout_ids168 = factual168[staycol168].drop_duplicates().tolist()[:100]
rollout_ids168 = [x.item() if isinstance(x, np.generic) else x for x in rollout_ids168]
print("\nNEXT MATCHED ROLLOUT REQUEST")
print(json.dumps({
    "type": "dose_edit", "model": "both", "split": "tuning",
    "stays": rollout_ids168
}))

print("\nRETAINED OBJECTS")
for name168 in [
    "lp_new136", "channel_weights160", "outer166",
    "backup_before167", "impulse_tests166", "impulse_results159"
]:
    if name168 in globals():
        obj168 = globals()[name168]
        print(name168, type(obj168).__name__,
              ("keys=" + str(list(obj168.keys()))) if isinstance(obj168, dict)
              else ("shape=" + str(getattr(obj168, "shape", None))))
        if name168 == "lp_new136" and isinstance(obj168, dict):
            for key168, value168 in obj168.items():
                if isinstance(value168, dict):
                    print(" ", key168, "keys:", list(value168.keys()))
                elif np.isscalar(value168):
                    print(" ", key168, value168)

notes168 = pathlib.Path("/work/notes.md").read_text()
print("\nEARLY IDENTIFICATION NOTES")
print(notes168[:11000])

with open("/work/notes.md", "a") as log168:
    log168.write(
        "\n\n## Run 168\n"
        "GPU factual pass 12 (/work/fwd_13), current scale-2 connection edit: "
        "original NLL 1.59164, edited 1.59182, delta +0.00018 on 100 tuning stays. "
        "Recovered matching rollout IDs and architecture/identification context. "
        "No weights changed. Free-rollout efficacy and broader factual stability remain untested.\n"
    )
```
