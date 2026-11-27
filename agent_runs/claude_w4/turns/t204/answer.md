The minimum-norm δ_I has norm 13.2 per mg/dL. It moves finger and lab predictions by +1 each with about 7% spread across bundles, and leaves both MAP predictions essentially unchanged. δ_N needs 28.7 per mmHg. So injecting the glucose effect through the prelude output is feasible with modest perturbations.

Next is the carrier. I'll choose two directions inside head 1's output space (insulin rate, NE rate) along which the natural slot residual barely varies. Then I'll fit ridge readouts b_I and b_N on the attention-weighted, norm1-normalised tokens each slot actually receives through head 1. These should return the insulin or NE rate at drug bundles and zero at measurement bundles.

The test is precision: readout errors must be well below 0.25 units/h for insulin and about 0.02 for NE, so the gates and dose terms are clean.

```python
kvW = load('core.prelude.0.attn.kv.weight').numpy(); WoP = load('core.prelude.0.attn.out.weight').numpy()
Wo1 = WoP[:, 48:96]
Cx = np.cov(x1all.T)
Ag = Wo1.T @ Cx @ Wo1; Bg = Wo1.T @ Wo1
evg, avg = scipy.linalg.eigh(Ag, Bg)
print('carrier var per unit norm (smallest 4):', np.round(evg[:4],5), ' vs median dir var %.2f' % np.median(np.diag(Cx)))
aI_ = avg[:,0]/np.linalg.norm(Wo1@avg[:,0]); aN_ = avg[:,1]/np.linalg.norm(Wo1@avg[:,1])
cI_dir = Wo1@aI_; cN_dir = Wo1@aN_
print('cos(cI,cN) %.3f' % (cI_dir@cN_dir))
L1T = L1(TOK).numpy(); L1Tn = L1(TOKn).numpy()
attd = r2['att'][:,1,:4,:].numpy(); attn_ = r2n['att'][:,1,:4,:].numpy()
Fd = np.einsum('njt,ntd->njd', attd, L1T).reshape(-1,192); Fn = np.einsum('njt,ntd->njd', attn_, L1Tn).reshape(-1,192)
rN = np.nan_to_num(RN_d.astype(float))
tI = np.repeat(rI,4); tN = np.repeat(rN,4)
Xf = np.r_[Fd, Fn]; wts = np.r_[np.ones(len(Fd)), 10*np.ones(len(Fn))]
for lam in [1e-6, 1e-4, 1e-2]:
    out_ = []
    for t_ in [tI, tN]:
        tt = np.r_[t_, np.zeros(len(Fn))]
        sw = np.sqrt(wts)
        A_ = Xf*sw[:,None]; b_ = tt*sw
        bb = np.linalg.solve(A_.T@A_ + lam*np.eye(192), A_.T@b_)
        e_ = Xf@bb - tt
        ed = e_[:len(Fd)].reshape(-1,4); en_ = e_[len(Fd):]
        out_.append('|b| %.0f maxerr drug %.4f rms %.4f | meas max %.4f' % (np.linalg.norm(bb), np.abs(ed).max(), np.sqrt((ed**2).mean()), np.abs(en_).max()))
    print('lam', lam, ' INS:', out_[0], ' NE:', out_[1])
```