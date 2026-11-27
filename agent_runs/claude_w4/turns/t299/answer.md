The record regression points to a leak rather than a wrong sign:

- **Drift without insulin:** stays given no insulin after hour 12 drift up just as much as the rest, +25 mg/dL by the end. Overall the drift grows by about +2 mg/dL per hour whatever the dose.
- **Dose-dependent part has the right sign:** the drift's dependence on the centred insulin signal is negative (−0.6), and the x2 shift falls with insulin given (−0.42 per unit). Both are much weaker than they should be.
- **What masks it:** a large positive offset (+7.6) in the shift hides the correct-signed part.
- **Truth on these stays:** about −13.2 for insulin x2 and +4.2 for NE x2. The model gives +3.3 and +0.23.
- **NE:** the NE channel shows the same pattern, a small leak and almost no dose response.

My leading suspect for the leak is the edited `b_in` input map. It reads a "channel" direction of the processed input that I chose to be quiet at drug bundles. Reading bundles (glucose/MAP) far outnumber drug bundles, and that direction may not be quiet there. A second, smaller suspect is a non-zero channel value at the "no drug" input token, which is present at every reading bundle.

Next run, entirely from stored arrays (no forward pass):
- compute the channel value the edit produces at the no-drug token;
- inspect which globals `build_v4` and the helper functions use, so I can rebuild with corrected directions;
- check, if possible, the channel coordinate at reading bundles from the stored reading-bundle tokens.

How to read the outcome:
- If the no-drug token gives a sizeable channel value, or the channel coordinate at reading bundles is far from zero, the leak is located and I can design it out.
- If both are about zero, the leak lies elsewhere, e.g. in removing the natural response along the channel. Then a forward test with partial edits is needed.

```python
import numpy as np, dis
f_const = Minv @ (A.T @ (Dkv @ LN1c))
print('channel readout f at no-drug token u_const [rI, (rN-.52)*pres, presI]:', f_const.round(4))
for fn in ['build_v4','prelude','prelude_b','tgt_from','solve_dc','solve_h','dout_of_dm','uv','zb','L','enc','mlp','coda','LN1','lnw']:
    c = globals()[fn].__code__
    print(fn, 'args', c.co_varnames[:c.co_argcount], 'globals', [n for n in c.co_names if n in globals()][:30])
for fname in ['/work/chan.npz','/work/v4_parts.npz','/work/rep_drug.npz','/work/v5_parts.npz','/work/dirs265.npz']:
    try:
        z = np.load(fname, allow_pickle=True); print(fname, {k: z[k].shape for k in z.files})
    except Exception as ex: print(fname, 'ERR', ex)
print('Tm', Tm.shape, 'Xs_m', Xs_m.shape, 'Td', Td.shape, ' norms Tm tokens:', np.linalg.norm(Tm[:5], axis=-1).round(1))
```