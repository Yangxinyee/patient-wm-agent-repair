The first run printed the architecture and the data README, and there is no notebook yet, so this is a fresh start. What I learned:
- Each infusion record lasts one hour at its rate.
- A dose edit scales the insulin or norepinephrine rate and dose from hour 12 on.
- The readout is the mean shift in glucose_finger (or map_arterial) over hours 13 to 24, edited minus factual.
- Clinicians chose the doses, so doses are confounded with the patient's condition.

Next I will start /work/notes.md and load the training records to see how they are structured: event counts, how often and when each drug is given, and the ranges of rates and readings. This tells me what variation exists for estimating the true drug effects. My working hypothesis is that dosing follows recent readings, perhaps with some random component. If dose varies substantially at a fixed recent reading, the records can be used to estimate the effect.

```python
import numpy as np, pandas as pd
notes = """# Notebook

## Run 1
- Read ARCHITECTURE.md and data README. Infusion records = 1 h at a rate. Dose edit scales insulin/norepi rate+dose from hour 12.
- Readout: mean shift of glucose_finger (insulin) / map_arterial (norepi) over hours 13-24, edited minus factual.
- Doses chosen by clinicians -> confounded. Need causal estimate from records.
"""
open('/work/notes.md','w').write(notes)

tr = pd.read_csv('/data/train.csv')
tu = pd.read_csv('/data/tuning.csv')
print(tr.shape, tu.shape)
print(tr.head(20).to_string())
print(tr.dtypes)
print(tr['event'].value_counts())
print("stays:", tr.stay_id.nunique(), tu.stay_id.nunique())
for ev in ['insulin','norepinephrine']:
    d = tr[tr.event==ev]
    print(ev, d[['rate','dose','hour']].describe().T.to_string())
    print(d.rate_unit.value_counts().to_dict(), d.dose_unit.value_counts().to_dict())
for ev in ['glucose_finger','glucose_lab','map_arterial','map_cuff']:
    d = tr[tr.event==ev]
    print(ev, d[['value','hour']].describe().T.to_string())
```