Each states file has 4,356 bundles for 100 tuning stays, with flags for insulin and norepinephrine. Pass 5 compares the original model with one where the dose input columns were scaled x5. That makes it a controlled perturbation of only the drug inputs.

Next I will measure the state change edited minus original:
- at insulin-only bundles, how large it is per block (out, the four h slots, z) relative to the natural spread of the original states;
- whether its principal direction is selective, meaning a direction where the dose-induced change is large compared with natural variation at reading bundles;
- how much of it persists at the reading bundles that follow.

A dose-dedicated direction in h would show as a high ratio of dose-induced to natural variance along the top difference direction. A ratio well below 1 would mean any state-space amplification also amplifies ordinary dynamics.

```python
zo = np.load('/work/fwd_5/states_original.npz', allow_pickle=True); ze = np.load('/work/fwd_5/states_edited.npz', allow_pickle=True)
Xo, Xe = zo['x'].astype(np.float64), ze['x'].astype(np.float64)
hi_, hn_, na_ = zo['has_insulin'].astype(bool), zo['has_norepinephrine'].astype(bool), zo['n_atoms']
sid5, hr5 = zo['stay_id'], zo['hour']
drug = hi_|hn_
print('bundles', len(Xo), 'ins only', (hi_&~hn_).sum(), 'ne only', (hn_&~hi_).sum(), 'both', (hi_&hn_).sum(), 'nondrug', (~drug).sum())
D5 = Xe - Xo
blocks = {'out':slice(0,192),'h0':slice(192,384),'h1':slice(384,576),'h2':slice(576,768),'h3':slice(768,960),'z':slice(960,992)}
nat_sd = Xo[~drug].std(0) + 1e-9
for nm_b, sl in blocks.items():
    for lab, msk in [('ins',hi_&~hn_),('ne',hn_&~hi_),('read',~drug)]:
        r = np.sqrt((D5[msk][:,sl]**2).mean(0)) / nat_sd[sl]
        print(nm_b, lab, 'rms dDelta/natsd: median %.3f max %.3f' % (np.median(r), r.max()), end=' | ')
    print()
# selectivity of top direction of Delta at insulin bundles (h slots stacked)
H = slice(192,960)
U_, S_, Vt_ = np.linalg.svd(D5[hi_&~hn_][:,H] - 0, full_matrices=False)
print('sv frac top5', np.round(S_[:5]**2/ (S_**2).sum(),3))
v1 = Vt_[0]
for lab, msk in [('ins',hi_&~hn_),('ne',hn_&~hi_),('read',~drug)]:
    print(lab, 'Delta proj sd %.4f mean %.4f | natural proj sd %.4f' % ((D5[msk][:,H]@v1).std(), (D5[msk][:,H]@v1).mean(), (Xo[msk][:,H]@v1).std()))
```