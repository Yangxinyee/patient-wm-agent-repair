The stored impulse responses are as follows.
- **Insulin:** about −1.9, −2.3, −2.0, −1.6 and −1.2 mg/dL per unit/h at lags 1–5. The effect then fades to about −0.2 by lags 10–12, so it is transient.
- **Norepinephrine:** about +2.6, +2.4, +1.3 and +1.0 mmHg per mcg/kg/min at lags 1–4, then roughly 0.

These are the targets. Next I will compute the true readouts these imply on the same tuning stays and rows that the dose_edit tool uses (fwd_1). That gives a concrete number for each of the four edits to calibrate against. If insulin x2 comes out near −17 and x1/2 near +8, that confirms my notes. The norepinephrine numbers I have not yet computed; I expect them to be small and positive for x2.

```python
import os, glob
print(os.listdir('/work/fwd_1'))
de1 = pd.read_csv('/work/fwd_1/dose_edit_original.csv')
print(de1.shape, list(de1.columns)[:14])
print(de1.head(3).to_string()[:600])
```