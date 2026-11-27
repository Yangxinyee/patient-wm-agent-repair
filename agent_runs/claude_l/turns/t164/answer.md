The vocab explains the 15 types: 5 event kinds × 3 atom kinds, so a glucose_finger WORLD measurement is type 1×3+0 = 3 or 0×5+1 = 1, depending on the ordering. It does not say what the 8 `last_numeric` inputs are, so I'll fit those against the tool's per-reading predicted means. First I need the column layout of the factual CSV (fwd_4) and the dose_edit CSV (fwd_1). I'll print their headers and two rows each.

```python
import pandas as pd
f4 = pd.read_csv('/work/fwd_4/factual_original.csv'); print(f4.shape); print(f4.head(3).to_string())
f1 = pd.read_csv('/work/fwd_1/dose_edit_original.csv'); print(f1.shape); print(f1.head(3).to_string())
print(json.load(open('/work/fwd_1/summary.json')))
```