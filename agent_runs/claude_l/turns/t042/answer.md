The last run printed the architecture description. It confirms that a dose edit reaches the rest of the model only through the drug atoms' rate and dose features. Those feed `encode_action` to give u_k, and u_k enters the posterior `z_post` and the prelude token.

This settles how the edit has to work. Under teacher forcing, the dose adds no information beyond the recorded history, and my negative control showed this. A perfect model would therefore respond to the dose only causally. This model responds about 4 times too weakly. The likely reason is that its history pathway has absorbed the policy-expected part of the dose effect. If so, simply amplifying the dose input would double-count that part on factual records and damage the factual fit. A correct correction must act on the dose relative to what the policy would have given.

Next I want to recover what forward pass 2 contained, since my notes do not say. I also want to recover the state of my local reimplementation before spending more forward passes.

```python
import glob
for f in sorted(glob.glob('/work/**', recursive=True)):
    if os.path.isfile(f): print(f, os.path.getsize(f))
print(open('/work/notes.md').read()[-1500:])
for fn in glob.glob('/work/fwd_2/*'):
    if fn.endswith('.npz'):
        Z = np.load(fn, allow_pickle=True); print(fn, {k: Z[k].shape for k in Z.files})
    elif fn.endswith('.csv'):
        print(fn); print(pd.read_csv(fn).head())
    else:
        print(fn); print(open(fn).read()[:1500])
print(json.load(open('/work/batches.json')).keys())
```