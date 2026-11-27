The original model's dose answers are all essentially zero:

| Edit | Predicted shift | Coverage |
|---|---|---|
| insulin ×2 → glucose_finger | +0.14 mg/dL | 917 stay-hours |
| insulin ×½ → glucose_finger | −0.08 mg/dL | 917 stay-hours |
| norepinephrine ×2 → map_arterial | +0.006 mmHg | 672 stay-hours |
| norepinephrine ×½ → map_arterial | +0.015 mmHg | 672 stay-hours |

The norepinephrine readout covers only 60 stays because it needs map_arterial, which only arterial-line stays have. So the model has learned almost no dependence of future readings on dose magnitude. That is not the pattern I guessed (a wrong sign). Whether it is wrong depends on what the drugs really do, which I must now estimate from the records.

Next I will inspect the per-stay-hour output file, to know exactly which hours count as "reached" and how the readout is built, and I will write my first notes. That lets me reproduce the readout with my own estimates of the true effect on the same stays.

```python
import glob
f = pd.read_csv('/work/fwd_1/dose_edit_original.csv')
print(f.shape); print(f.columns.tolist())
print(f.head(30).to_string())
print(f.describe().round(3).to_string())
with open('/work/notes.md','a') as fh:
    fh.write("""
## Findings so far (runs 1-11, fwd 1)
- All stays 25.17h; readings at integer hours (-1..24), drugs at h+5min (0..24), each 1h.
- Insulin policy = sliding scale on finger reading at that hour: <150:0, 150-180:~1.05, 180-220:~2.1, 220-260:~3.13, 260-300:~4.19, >=300:~6.2 u/h,
  multiplicative noise CV~0.33, rounded to 0.5; ~3% omitted. Insulin almost only given when finger reading present (98%).
- NE policy: titration. Start prob high at MAP<56, ~0 above 68. With NE running: log ratio +0.05..0.13 if MAP<64, ~0 for 64-76, -0.27 if MAP>76. sd 0.3. Cap at 1.0 (16% of records at cap). NE dose = 4.8*rate.
- NE x2 pushes rate to 2.0 (z=2.37, bin 16) - outside training range (max z 0.76).
- fwd1 (original, tuning, 100 stays w/ drugs >=12): ins x2 +0.141, ins x1/2 -0.084, NE x2 +0.006, NE x1/2 +0.015 => model has ~no dose effect.
  NE readout only on 60 stays with map_arterial.
""")
```