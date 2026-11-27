The sel1 targets I just recomputed don't match the notebook's (−13.99, 6.99, 4.85, −2.43), so this run has a bug. It is in the NE kernel: `gam_ne_clean` in effects.json already starts at lag 0 (9 values beginning with 0), and I prepended another 0. That shifted the NE effect one hour late, which is why both NE targets came out smaller.

The insulin targets (−14.35 and +7.17) also differ from the notebook. They may have been computed with the separate /work/beta_ins.json estimate rather than effects.json's `beta_ins`.

This matters for the main readout, so this run:
- recomputes the targets with the correctly aligned NE kernels (clean and old);
- compares both insulin kernels;
- checks which combination reproduces the notebook's sel1 numbers.

Whichever reproduces them identifies the estimate behind the calibration. I'll then judge whether −13.9 on sel1 and −13.4 on sel2 are on target.

```python
import numpy as np, json
bj = json.load(open('/work/beta_ins.json')); print(type(bj).__name__, (list(bj.keys()) if isinstance(bj, dict) else len(bj)))
bj_ = np.asarray(bj if isinstance(bj, list) else (bj.get('beta_ins') or bj.get('beta') or list(bj.values())[0]), float).ravel()
print('beta_ins.json', np.round(bj_,2)); print('effects beta', np.round(bi,2))
print('gam_ne old', np.round(np.asarray(eff['gam_ne'],float),2)); print('gam clean', np.round(gn,2))
def tg_only(path, fns):
    d = pd.read_csv(path); d = d[d.reached]; out = {}
    for ed, fac in [('insulin_x2',2.0),('insulin_half',0.5),('ne_x2',2.0),('ne_half',0.5)]:
        x = d[d.edit==ed]; f_ = fns['ins'] if ed.startswith('ins') else fns['ne']
        out[ed] = round(float(np.mean([f_(int(s),int(h),fac) for s,h in zip(x.stay_id, x.hour)])),2)
    return out
gnc = np.r_[gn, np.zeros(13-len(gn))]; gno = np.asarray(eff['gam_ne'], float)
for lab, bb_, gg_ in [('eff beta + clean', bi, gnc), ('json beta + old', bj_, gno)]:
    fns = {'ins': lambda s,h,f,b=bb_: true_shift_ins(s,h,f,b), 'ne': lambda s,h,f,g=gg_: true_shift_ne(s,h,f,g)}
    print(lab, 'sel1', tg_only('/work/fwd_13/dose_edit_edited.csv', fns), 'sel2', tg_only('/work/fwd_14/dose_edit_edited.csv', fns))
```