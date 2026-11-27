Run 92 settles the structure. The value head is an MLP (192→192→17) reading `with_code` through `query_norm`, which has an affine gain I cannot see. `encode_action` ends in an affine norm, and the prelude normalises every token again (`norm1`) before attention. So any extra dose signal in u_k or in the z token is renormalised twice. That explains why scaling the dose inputs saturated: it only rotates the token, it cannot make it larger.

A linear estimate from passes 1, 4 and 9 says the numeric dose features contribute only about -0.17 of the -1.85 insulin x2 response; the dose bins carry the rest. Insulin and norepinephrine also share the only dose-carrying z dimension (z26), so a remap through z cannot separate the two drugs. I am dropping the z route.

The one coordinate system I can observe is the residual stream h (192 per slot). `b_in`'s output and `c_out`'s input and output all live in it, up to the coda.

Next I want to measure, from saved pass-5 states (original versus dose-column x5), where the dose-induced state change lives in h and out. I also want its size relative to natural variation and how it decays over the following bundles. A clearly dose-dedicated direction would show up as a large ratio of dose-induced to natural variance along one direction. A diffuse change would rule out direction-selective edits in the state space.

```python
import numpy as np, json
for k in ['fwd_3','fwd_5']:
    for fn in sorted(os.listdir('/work/'+k)):
        if fn.endswith('.npz'):
            zf = np.load(f'/work/{k}/{fn}', allow_pickle=True)
            print(k, fn, {kk: zf[kk].shape for kk in zf.files})
    print(k, open(f'/work/{k}/summary.json').read()[:300])
cfg = VOCAB.get('config', {})
print({kk: cfg[kk] for kk in list(cfg)[:40]})
```