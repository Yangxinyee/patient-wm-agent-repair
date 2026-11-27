The original model predicts:

| edit | predicted mean shift | stay-hours | stays |
|---|---|---|---|
| insulin ×2 → glucose_finger | −2.1 mg/dL | 797 | 84 |
| insulin ×½ → glucose_finger | +1.3 mg/dL | 797 | 84 |
| NE ×2 → map_arterial | +1.1 mmHg | 626 | 56 |
| NE ×½ → map_arterial | −0.7 mmHg | 626 | 56 |

The signs are right but the magnitudes look far too small compared with my rough record-based expectation (−10 to −20 mg/dL and a few mmHg). This fits a model that learned the immediate one-hour effect only weakly, or that partly cancels it with dose-as-indication associations.

Next I'll read the per-stay-hour CSV to learn exactly which rows count (the "reached" rule), and compute my impulse-response estimate of the true shift for exactly those rows. That gives a like-for-like comparison. If my estimate is several times larger than the model's, the edit must strengthen the dose pathways substantially.

```python
import glob
print(glob.glob('/work/fwd_1/*'))
de = pd.read_csv('/work/fwd_1/dose_edit_original.csv')
print(de.shape); print(de.columns.tolist()); print(de.head(30).to_string())
```