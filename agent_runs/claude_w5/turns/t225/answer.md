The edit behaves as designed:
- The gate is clean (drug ≥ 0.87, other bundles ≤ -0.78).
- Each carrier reaches only its own quiet direction (no cross-gain).
- At non-drug bundles h_new moves by only about 0.1 in norm, roughly 0.04 mg/dL in predictions, which is negligible.
- At drug bundles the realized injection averages about 0.84 of the intended one (sd 0.47 across bundles). The spread comes from the LayerNorm scale of P, which affects the per-bundle gain but not the pooled readout.

Next, two local checks before spending forward passes:
1. A teacher-forced pass of the edited core over all 100 training stays, with local head NLL (4 prior draws, same seed) for edited versus original, overall and by reading type.
2. Regressions of the next-reading prediction shifts on the decoded residuals.

The edit is safe if the NLL change is near zero or slightly negative, well inside the 0.005 budget. The response slopes should come out around -1.2 (glucose per insulin residual) and +1.0 (MAP per norepinephrine residual), with small cross-terms.

```python
print('H target sample', [H[i]['target'] for i in range(3)], type(H[0]['target']).__name__)
def tf_states(W_, s):
    global WCUR
    WCUR = W_; st = steps_all[s]; o_, h_ = None, None; outs = []
    for k, d in enumerate(st):
        d2 = d if k == 0 else dict(d, out0=o_, h0=h_)
        r_ = mystep(d2); o_, h_ = r_[0], r_[1]; outs.append((o_, h_))
    WCUR = W64; return outs
def stay_views(s, outs):
    V, idx = [], []
    for k in range(1, len(bundles_B[s])):
        for i in Hidx.get((s, k), []):
            dt = (bundles_B[s][k][0] - bundles_B[s][k-1][0]) * 3600.0
            V.append(view0(outs[k-1][0], outs[k-1][1], dt)); idx.append(i)
    return V, idx
VO, VEd, IDX = [], [], []
t0_ = time.time()
for s in sids_B:
    o_orig = [(d['res'][0], d['res'][1]) for d in steps_all[s]]; o_ed = tf_states(WE, s)
    v1, i1 = stay_views(s, o_orig); v2, _ = stay_views(s, o_ed)
    VO += v1; VEd += v2; IDX += i1
VO = torch.stack(VO); VEd = torch.stack(VEd); IDX = np.array(IDX)
tgt_ = torch.tensor([int(H[i]['target']) for i in IDX])
codes_ = np.array([H[i]['code'] for i in IDX])
def nll_of(V):
    p = probs_from_v0(V, fixed[IDX], TRr[IDX], nz=4, seed=0); p = torch.as_tensor(p)
    return -torch.log(p[torch.arange(len(IDX)), tgt_].clamp_min(1e-12)).numpy(), p
n0, p0 = nll_of(VO); n1, p1 = nll_of(VEd)
print('time', round(time.time()-t0_,1), 'rows', len(IDX), 'NLL orig', n0.mean().round(4), 'edited', n1.mean().round(4), 'diff', (n1-n0).mean().round(5))
for c in np.unique(codes_):
    m = codes_ == c; print('  ', c, m.sum(), 'diff', (n1[m]-n0[m]).mean().round(5))
mean0 = (p0[:,1:]*cen[IDX]).sum(1)/p0[:,1:].sum(1); mean1 = (p1[:,1:]*cen[IDX]).sum(1)/p1[:,1:].sum(1)
dm = (mean1-mean0).numpy()
# response at reading right after drug bundles
xs_I, xs_N, dg, dmp = [], [], [], []
pos = {i: j for j, i in enumerate(IDX)}
for r in DB:
    d = steps_all[r['sid']][r['k']]; R = {}; mystep(d, R=R); n3 = R['n3'][0].numpy()
    xI, xN = n3@wxI + bxI, n3@wxN + bxN
    for i in Hidx.get((r['sid'], r['k']+1), []):
        if H[i]['code'] == 'glucose_finger': xs_I.append((xI, xN)); dg.append(dm[pos[i]])
        if H[i]['code'] == 'map_arterial': xs_N.append((xI, xN)); dmp.append(dm[pos[i]])
A = np.c_[np.array(xs_I), np.ones(len(xs_I))]; cg = np.linalg.lstsq(A, np.array(dg), rcond=None)[0]
A2 = np.c_[np.array(xs_N), np.ones(len(xs_N))]; cm = np.linalg.lstsq(A2, np.array(dmp), rcond=None)[0]
print('next glucose shift ~ xI, xN, 1:', cg.round(3), '| next MAP shift ~ xI, xN, 1:', cm.round(3))
```