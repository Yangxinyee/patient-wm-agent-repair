The kernel ran out of memory and restarted, so every variable and helper function from earlier runs is gone. The saved edit is still the doubled-dose emulation, which must not be submitted.

This run will:
- drop that edit;
- rebuild only the basic pieces I need;
- record the crash in the notebook;
- list the head tensors so q can be rebuilt later if needed.

It will also answer the key feasibility question without the lost helpers. In fwd 12 I will look at the model's summary vector `out` just before each glucose or MAP reading, and see how much it moves under doubled doses compared with its natural variation (GLS signal-to-noise ratio per unit dose²). `out` is the dominant changing part of the view that the value head reads, so this is a good proxy.

If the SNR is around 1 or more, a gated value-head readout is worth rebuilding in the remaining runs. If it is far below that, the dose information is lost before the readings, and I will stop and keep the original weights.

```python
import numpy as np, pandas as pd, json, torch, gc
TT = lambda x: x.numpy() if hasattr(x, 'numpy') else np.asarray(x)
for n in list(edited()): drop_edit(n)
print('edited after drop:', edited())
print([(n, tuple(SHAPES[n])) for n in NAMES if not n.startswith(('core.', 'encode', 'embed.', 'z_post', 'z_in'))])
SO = np.load('/work/fwd_12/states_original.npz'); SE = np.load('/work/fwd_12/states_edited.npz')
xo = SO['x'].astype(np.float64); xe = SE['x'].astype(np.float64)
sid = SO['stay_id']; hr = SO['hour'].astype(float); hi = SO['has_insulin'].astype(bool); hn = SO['has_norepinephrine'].astype(bool)
print('state hours sample', hr[:6], 'n rows', len(hr))
ids = [int(i) for i in json.load(open('/work/sel_states.json'))]
parts = []
for ch in pd.read_csv('/data/train.csv', usecols=['stay_id', 'hour', 'event', 'value', 'rate'], chunksize=1_000_000):
    parts.append(ch[ch.stay_id.isin(ids)])
d99 = pd.concat(parts); del parts; gc.collect()
print(d99.head(4).to_string()); print('rows', len(d99))
rows_by_stay = {s: np.where(sid == s)[0] for s in ids}
rd = d99[d99.event.isin(['glucose_finger', 'glucose_lab', 'map_arterial', 'map_cuff'])].copy()
jj = []
for s, h in zip(rd.stay_id.values, rd.hour.values.astype(float)):
    r = rows_by_stay[int(s)]; i = r[np.argmin(np.abs(hr[r] - h))]; jj.append(i if abs(hr[i] - h) < 0.01 else -1)
rd['j'] = jj; print('readings matched', (rd.j >= 0).mean(), 'first-row readings', (rd.j == np.array([rows_by_stay[int(s)][0] for s in rd.stay_id])).sum())
rd = rd[(rd.j > 0)]; rd = rd[np.array([sid[j - 1] == s for j, s in zip(rd.j, rd.stay_id)])]
drug = d99[d99.event.isin(['insulin', 'norepinephrine'])]
mp = {nm: {(int(s), int(np.floor(h))): float(r) for s, h, r in drug[drug.event == nm][['stay_id', 'hour', 'rate']].itertuples(index=False)} for nm in ['insulin', 'norepinephrine']}
rateI = np.array([mp['insulin'].get((int(s), int(np.floor(h))), np.nan) if f else np.nan for s, h, f in zip(sid, hr, hi)])
rateN = np.array([mp['norepinephrine'].get((int(s), int(np.floor(h))), np.nan) if f else np.nan for s, h, f in zip(sid, hr, hn)])
print('rate match ins %.3f ne %.3f' % (np.isfinite(rateI[hi]).mean(), np.isfinite(rateN[hn]).mean()))
def lastrows(flag):
    l1 = np.full(len(sid), -1); l2 = np.full(len(sid), -1); c1 = c2 = -1; ps = None
    for b in range(len(sid)):
        if sid[b] != ps: c1 = c2 = -1; ps = sid[b]
        if flag[b]: c2 = c1; c1 = b
        l1[b] = c1; l2[b] = c2
    return l1, l2
lI1, lI2 = lastrows(hi); lN1, lN2 = lastrows(hn)
kk = rd.j.values - 1
fe = lambda l, rate: np.array([rate[l[k]] if l[k] >= 0 and np.isfinite(rate[l[k]]) else 0.0 for k in kk])
FD = np.stack([fe(lI1, rateI), fe(lI2, rateI), fe(lN1, rateN), fe(lN2, rateN)], 1)
ev = rd.event.values; isG = np.isin(ev, ['glucose_finger', 'glucose_lab']); isM = np.isin(ev, ['map_arterial', 'map_cuff'])
for nm, msk, col in [('INS@gluc', isG, 0), ('NE@map', isM, 2)]:
    for sp, sl in [('out', slice(0, 192)), ('slotmean', None)]:
        Xo = xo[kk[msk], :192] if sl is not None else xo[kk[msk], 192:960].reshape(-1, 4, 192).mean(1)
        Xe = xe[kk[msk], :192] if sl is not None else xe[kk[msk], 192:960].reshape(-1, 4, 192).mean(1)
        Y = Xe - Xo; Fm = FD[msk]; Bc = np.linalg.lstsq(Fm, Y, rcond=None)[0]; fit = Fm @ Bc; a = Bc[col]
        Sg = np.cov(Xo.T) + 1e-6 * np.trace(np.cov(Xo.T)) / 192 * np.eye(192); snr = float(a @ np.linalg.solve(Sg, a))
        print(nm, sp, 'n %d |dY| %.4f fitR2 %.3f |a1| %.4f |a2| %.4f natural tr %.3f SNR/unit2 %.3f noise sd %.3f dose units' % (msk.sum(), np.sqrt((Y**2).sum(1)).mean(), 1 - ((Y - fit)**2).sum() / (Y**2).sum(), np.linalg.norm(a), np.linalg.norm(Bc[col + 1]), np.trace(Sg), snr, 1 / np.sqrt(max(snr, 1e-12))))
with open('/work/notes.md', 'a') as f:
    f.write('- run283: KERNEL OOM RESTART (all vars/helpers lost). run284: dropped emulation edit (model = original). Lesson: never rebind single-letter globals; read train.csv in chunks.\n')
```