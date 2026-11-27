Run 1 gave me the architecture and the data columns. What matters most:
- Doses enter the model only through the insulin and norepinephrine atom embeddings: `embed.numeric` z_dose/z_rate columns and `embed.bins.dose/rate`.
- Those atoms pass through `encode_action` to give u_k, which feeds `z_post` and the prelude.
- The rollout keeps the recorded bundle structure, hides readings from hour 13, and has the model sample its own values.

My working hypothesis is about where confounding could come from. Clinicians chose doses using information the record may not fully show, such as noisy readings versus the true state. So the model may read a dose as evidence about the patient's state rather than as a cause.

Before going further I need to know what the records look like. That means event counts, timing, dose distributions, and how doses relate to recent readings. This tells me how to estimate true effects; there is no hypothesis test yet. First I'll start the notebook and load the data.

```python
import pandas as pd, numpy as np, time
notes = """# NOTES (task: repair ICU world model dose effects)
## Run log
- Run1: read ARCHITECTURE.md, README. Doses enter only via insulin/norepi atom embeddings (embed.numeric z_dose/z_rate cols, embed.bins.dose/rate) -> encode_action -> u_k -> z_post & prelude.
"""
open('/work/notes.md','w').write(notes)
t0=time.time()
tr = pd.read_csv('/data/train.csv')
tu = pd.read_csv('/data/tuning.csv')
print(time.time()-t0)
print(tr.shape, tu.shape)
print(tr.head(30).to_string())
print(tr.event.value_counts())
print(tr.dtypes)
```