The architecture is clear. Insulin and norepinephrine atoms are actions. Their rate and dose enter only through the atom embedding (`embed.numeric` z_dose/z_rate columns, `embed.bins.dose`/`embed.bins.rate`). They then pass through `encode_action` into u_k, and from there into the z posterior and the prelude. Readings are predicted from value bins through `attr_heads.value_bin` plus a transition table keyed on the last value bin. A dose edit therefore acts only through the action pathway u_k. That is the natural place for an edit.

Next I will load the training records and look at their structure: event counts, timing of readings and infusions, and how rates are distributed. This is descriptive and tests no hypothesis yet. I need it to design the estimate of the true drug effects.

```python
import pandas as pd, numpy as np
with open('/work/notes.md', 'a') as f:
    f.write("""- Architecture: dose/rate enter only via insulin/norepi atom embedding (embed.numeric cols z_dose,z_rate; embed.bins.dose/rate),
  -> encode_action -> u_k -> z_post and prelude tokens. Values predicted via attr_heads.value_bin + embed_transition.value_bin[code*17+lastbin].
  Dose edit: doses x2 / x0.5 from hour 12, readings from h13 drawn by model; readout = mean pred glucose_finger (or map_arterial) edited-factual over h13-24.
""")
tr = pd.read_csv('/data/train.csv')
print(tr.shape); print(tr.head(20).to_string())
print(tr.event.value_counts())
print(tr.groupby('event')[['value','rate','dose']].describe().T.to_string())
```