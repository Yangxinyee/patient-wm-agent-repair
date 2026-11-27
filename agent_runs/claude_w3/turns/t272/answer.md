I have local helpers for the prelude and bundle encoding (`prelude_attn`, `enc_bundle`, `zpost_mean`). Next I'll check what the saved states files hold and how well the local posterior mean matches the tool's on dose-only bundles. That tells me whether the local dose encoding is reliable enough to design the write-in weights.

```python
for k in [3, 4]:
    fs = glob.glob(f'/work/fwd_{k}/*.npz')
    for x in fs:
        z = np.load(x, allow_pickle=True)
        print(k, os.path.basename(x), {kk: z[kk].shape for kk in z.files})
```