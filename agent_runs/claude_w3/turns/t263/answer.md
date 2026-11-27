Dimensions 49 and 185 have been set aside as dedicated channels: every layer's writes to them are zeroed, and every norm's weight and bias there is zero. `b_in` and `c_out` were changed throughout, which I can't explain yet. I'll look at forward passes 5 and 6 (which I don't remember) and at the saved arrays, to see whether this edit was ever run on the tool.

```python
import glob
for k in [5, 6]:
    fs = sorted(glob.glob(f'/work/fwd_{k}/*'))
    print(k, [os.path.basename(x) for x in fs])
    for x in fs:
        if x.endswith('.json') or x.endswith('.txt') or x.endswith('.md'):
            print(open(x).read()[:600])
        elif x.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(x); print(os.path.basename(x), df.shape, list(df.columns)[:12]); print(df.head(3).to_string()[:800])
for f in ['/work/arrays.npz', '/work/theta.npz']:
    z = np.load(f, allow_pickle=True); print(f, {k: z[k].shape for k in z.files})
```