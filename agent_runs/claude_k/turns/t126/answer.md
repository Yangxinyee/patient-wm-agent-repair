`out_` exactly matches the recorded state after the preceding action bundle (max difference 0), so Q99 and my per-reading arrays are built from the right state. The weak action-B signal at the readings is real, not a reconstruction artefact.

So the head can't fix action B, and any real fix must write the dose into the state. That needs an exact model of how the action atoms enter the posterior. I'll give the exact reconstruction one more bounded try, testing the cheapest structural variants on the 52 single-start bundles:
- z_post input order [ctx, u, e] instead of [ctx, e, u];
- e for action-only bundles taken as the set encoder of an empty set rather than zero;
- the bin convention (≤ versus <).

An error near zero settles the conventions. If all variants stay around 0.09, I'll stop reconstructing and switch to an empirical upstream edit calibrated with the tool.

```python
def zmean_o(ctx, e, u, order):
    x = np.concatenate([ctx, e, u] if order == 'eu' else [ctx, u, e]); return mlp_(x, 'z_post')[:32]
def e_empty():
    s = lin_(np.array([np.log(1.0)]), 'encode.size'); return LN(mlp_(s, 'encode.mlp2'), P['encode.norm.weight'], P['encode.norm.bias'])
def run_var(order, efn, le):
    E = []
    for j, rate in rows:
        dt = (hrS[j] - hrS[j - 1]) * 3600; ctx = ctx_from(xs[j - 1], dt)
        v = atom_vec(8, kind=0, ekind=2, zd=zsc_('insulin', 'dose', rate), hd=1.0, zr=zsc_('insulin', 'rate', rate), hr=1.0,
                     dbin=fb_('insulin', 'dose', rate, le), rbin=fb_('insulin', 'rate', rate, le))
        E.append(np.abs(xs[j, 960:992] - zmean_o(ctx, efn(), setenc(v, 'encode_action'), order)).mean())
    return np.mean(E)
for order in ('eu', 'ue'):
    for en, efn in [('e0', lambda: np.zeros(192)), ('eEmpty', e_empty)]:
        for le in (False, True):
            print(order, en, 'le' if le else 'lt', '%.4f' % run_var(order, efn, le))
```