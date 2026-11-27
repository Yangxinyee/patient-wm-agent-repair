The shapes show that the prelude attention has no biases, and that `encode_action` is two 192→768→192 MLPs followed by a LayerNorm. One clue stands out: numeric ×5 (−2.54 / +1.69) and head-0 ×4 (−2.53 / +1.67) are two very different edits with almost identical readouts. That points to a ceiling further downstream. Next I will compare their hour-by-hour insulin ×2 shift profiles with the original's.

```python
import glob
def prof(k,edit='insulin_x2'):
    out={}
    for f in sorted(glob.glob(f'/work/fwd_{k}/dose_edit_*.csv')):
        df=pd.read_csv(f); df=df[(df.edit==edit)&(df.reached.astype(bool))]
        out[f.split('_')[-1][:-4]]=df.groupby('hour').shift.mean().round(2).values
    return out
for k in [1,6,18]:
    print(k, prof(k))
```