The fixed directions behave consistently at the head:
- 10 units of v̄_I move the next glucose prediction by +3.18 ± 0.48 and MAP by only +0.07 ± 0.13.
- 10 units of v̄_N move MAP by +2.01 ± 0.31 and glucose by +0.05 on average (sd 1.1).
- My local head matches the forward tool's predicted means (corr 0.9998); views match H to 0.01.

So v̄_I and v̄_N serve as write directions.

Initial gains: make each extra unit of residual shift the next reading by the per-hour amount whose accumulation over hours 13–24 roughly gives the g-estimation pooled effect. That is about -1.46 mg/dL per u/h of insulin (κ_I ≈ -4.6) and +1.22 mmHg per mcg/kg/min of norepinephrine (κ_N ≈ +6.1). Forward passes will fine-tune these.

Now I assemble the full edit locally:
- three prelude-FFN units (twin-gated: gate gain 25, residual scale 0.5) write a carrier of 0.1·x along q_I and q_N;
- a rank-2 b_in edit turns the carrier into κ·x·v̄ in h_new and cancels b_in's original response to the carrier.

The edit is right if, at drug bundles, the realized change in h_new matches the intended κ·x·v̄ closely, it is near zero at non-drug bundles, and the two drug channels don't leak into each other.

```python
WCUR = W64
def Wt(n): return WCUR[n]
def linb(x, pre):
    y = x @ Wt(pre+'.weight').T
    b = WCUR.get(pre+'.bias')
    return y + b if b is not None else y
def ridge_all(X, y, lam):
    mu = X.mean(0); Xc = X - mu; ym = y.mean()
    w = np.linalg.solve(Xc.T@Xc + lam*np.eye(X.shape[1]), Xc.T@(y-ym)); return w, ym - mu@w
wxI, bxI = ridge_all(Xr, np.array(Yr['I']), 0.01); wxN, bxN = ridge_all(Xr, np.array(Yr['N']), 0.01)
wg, bg = ridge_all(Xg, yg, 0.01)
sg_all = Xg@wg + bg; print('gate (all-fit) drug min', sg_all[yg>0].min().round(3), 'non max', sg_all[yg<0].max().round(3), '|wg|', np.linalg.norm(wg).round(2))
def build_edit(kI, kN, Bg=25.0, g=0.5, rho=0.1, units=(502, 131, 140)):
    WE = dict(W64); jI, jN, jO = units
    W1 = W64['core.prelude.0.ffn.0.weight'].clone(); b1 = W64['core.prelude.0.ffn.0.bias'].clone(); W2 = W64['core.prelude.0.ffn.2.weight'].clone()
    T = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float64))
    W1[jI] = T(Bg*wg + g*wxI); b1[jI] = float(Bg*bg + g*bxI)
    W1[jN] = T(Bg*wg + g*wxN); b1[jN] = float(Bg*bg + g*bxN)
    W1[jO] = T(Bg*wg); b1[jO] = float(Bg*bg)
    cs = []
    for q in (q_I, q_N):
        c = q / gam_pn; c = c - c.mean(); cs.append(c)
    # carrier gain per unit c, averaged over sample drug bundles
    G = np.zeros((2, 2))
    for r in DB[:200:4]:
        d = steps_all[r['sid']][r['k']]; R0 = {}; mystep(d, R=R0)
        for a, c in enumerate(cs):
            R1 = {}; mystep(d, R=R1, dP=torch.from_numpy(np.tile(c, (4,1))))
            dE = (R1['E'] - R0['E']).numpy().mean(0)
            G[a, 0] += dE @ q_I; G[a, 1] += dE @ q_N
    G /= 50
    cs = [cs[0] * rho / G[0,0], cs[1] * rho / G[1,1]]
    W2[:, jI] = T(cs[0] / g); W2[:, jN] = T(cs[1] / g); W2[:, jO] = T(-(cs[0] + cs[1]) / g)
    WE['core.prelude.0.ffn.0.weight'] = W1; WE['core.prelude.0.ffn.0.bias'] = b1; WE['core.prelude.0.ffn.2.weight'] = W2
    Bin = W64['core.b_in.weight'].clone().numpy(); spd = F_.softplus(W64['core.delta_in']).numpy()
    for q, t in ((q_I, kI*vbar_I), (q_N, kN*vbar_N)):
        Bin = Bin + np.outer(t/rho/spd - Bin@q, q)
    WE['core.b_in.weight'] = torch.from_numpy(Bin)
    return WE, G
WE, Gc = build_edit(-4.6, 6.1)
print('carrier cross-gain matrix (per unit c, before scaling)', Gc.round(5))
# verify
def run_cmp(WE_, recs):
    global WCUR
    out = []
    for d, xI, xN in recs:
        WCUR = W64; h0_ = mystep(d)[5]
        WCUR = WE_; h1_ = mystep(d)[5]
        WCUR = W64
        out.append((h1_ - h0_).numpy())
    return out
recs = []
for r in DB[1::6]:
    d = steps_all[r['sid']][r['k']]; R = {}; mystep(d, R=R); n3 = R['n3'][0].numpy()
    recs.append((d, n3@wxI + bxI, n3@wxN + bxN))
dH = run_cmp(WE, recs)
err = []; 
for (d, xI, xN), dh in zip(recs, dH):
    tgt = np.tile(-4.6*xI*vbar_I + 6.1*xN*vbar_N, (4,1)); err.append(np.linalg.norm(dh - tgt) / (np.linalg.norm(tgt) + 1e-9))
gI_ = [float((dh.mean(0) @ vbar_I) / (vbar_I@vbar_I) / (-4.6*xI)) for (d, xI, xN), dh in zip(recs, dH) if abs(xI) > 0.2]
print('drug bundles: rel err median', np.median(err).round(3), 'p90', np.percentile(err, 90).round(3), '| realized/intended gain on vbar_I median', np.median(gI_).round(3), 'sd', np.std(gI_).round(3))
nd = [d for s in sids_B[::10] for d in steps_all[s] if (d['u'] - u_empty).norm() < 1e-6][:150]
dHn = run_cmp(WE, [(d, 0, 0) for d in nd])
print('non-drug |dh_new| max', max(np.abs(x).max() for x in dHn), 'mean', np.mean([np.linalg.norm(x) for x in dHn]))
```