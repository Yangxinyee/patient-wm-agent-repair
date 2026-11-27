Run 130 gave the value-head output rule: class 0 means "no value", classes 1–16 are fine bins, and the predicted mean is Σ p(b)·centre[b−1].
- Glucose bins are 10–20 mg/dL wide (73…272).
- MAP bins are 3–7 mmHg wide (33…101).
- The true insulin-×2 shift (about −14 mg/dL) is therefore about one bin, so an exponential logit tilt along the bin centres is a workable way to move the mean.
- `gen_samples(pert=True)` scales each dose by a random factor exp(U(−0.8, 0.8)), roughly ×0.45 to ×2.2, which covers the evaluation's ×½ and ×2. The perturbed sample set is therefore suitable for training decoders that respond to the dose itself.

Next I need the size of the missing direct effect, c_k, at each lag. I'll regress the replica's factual residuals (`res_all`, aligned with `keep`) on lagged rule innovations (`NUm` for insulin, `NUNm` for norepinephrine). First I'll check how these per-stay-hour matrices are indexed by matching `Dm`/`Rm` against the lag-1 doses stored in `keep`.

What would confirm the picture: clearly negative insulin coefficients on glucose and positive norepinephrine coefficients on MAP, mostly at lags 1–3, about the size of my true effects minus the model's weak ones. Coefficients near zero would mean the model's teacher-forced dose response is already right, and the problem lies in the rollout dynamics.

```python
codesK = np.array([t[0] for t in keep]); sidK = np.array([t[1] for t in keep]); hrK = np.array([t[2] for t in keep]).astype(float)
lagK = np.array([[t[5],t[6],t[7]] for t in keep], dtype=float)
id2ix = {sv: i for i, sv in enumerate(train_ids)}
ixK = np.array([id2ix.get(sv, -1) for sv in sidK])
print('missing ids', (ixK < 0).sum(), 'rs_all==ixK', np.mean(rs_all == ixK), 'rt_all==hr', np.mean(rt_all == hrK))
for nmM in ['Dm', 'Rm', 'dDm']:
    M_ = g[nmM]
    for cd_ in ['glucose_finger', 'map_arterial']:
        out_ = []
        for o in range(-1, 3):
            col = (hrK - 1 + o).astype(int); sel = (col >= 0) & (col < 26) & (codesK == cd_) & (lagK[:,0] > 0)
            out_.append(round(float(np.mean(np.isclose(M_[ixK[sel], col[sel]], lagK[sel,0], atol=1e-3))), 3))
        print(nmM, cd_, out_)
```