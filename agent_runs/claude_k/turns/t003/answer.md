The README confirms the layout. Each infusion record is one hour at a fixed rate, and "clinicians chose the doses." Doses respond to the patient's state, so the dose–reading association in the records is confounded. I'll need to estimate the drug effects with that confounding in mind, e.g. by adjusting for the pre-dose state or by finding exogenous variation in the doses.

The vocab shows the predicted mean is limited to bin centres. Glucose_finger runs 73–272 and map_arterial 33–101. The norepinephrine rate edges top out at 1.0, so x2 doses above 1.0 all land in bin 16, with only z_rate still changing.

Next I'll load the training records, look at their structure (events per stay, timing of readings and infusions, dose distributions), and start the notebook. Clean hourly infusion records with readings in between would confirm my picture. It would let me set up a regression of the next reading on the current dose, adjusted for the history.

```python
import pandas as pd, numpy as np
for e in V['numeric']:
    print(e['code'], e['attribute'], e['median'], e['iqr'], e['fine_bin_edges'])
tr = pd.read_csv('/data/train.csv')
tu = pd.read_csv('/data/tuning.csv')
print(tr.shape, tu.shape)
print(tr.head(30).to_string())
print(tr.event.value_counts())
print(tr.groupby('stay_id').size().describe())
```