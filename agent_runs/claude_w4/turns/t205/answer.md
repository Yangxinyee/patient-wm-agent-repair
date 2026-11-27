The carrier directions in head 1's output space have natural variance of only about 0.04 to 0.08 per unit norm, against about 250 for a typical direction. A moderate-size carrier is therefore read back with small noise.

With almost no regularisation the attention-weighted readouts are good on average (rms error 0.016 for the insulin rate and 0.012 for the NE rate), but a few bundles have maximum errors of 0.25 and 0.08. That is too much for gating NE at very small rates. A separate presence indicator (0/1) would make a more robust gate than the rate itself.

I'll locate the worst errors by bundle type and test presence readouts for insulin and NE.

```python
def fitb(t_, lam=1e-6, wmeas=10.0):
    tt = np.r_[t_, np.zeros(len(Fn))]; sw = np.sqrt(np.r_[np.ones(len(Fd)), wmeas*np.ones(len(Fn))])
    A_ = Xf*sw[:,None]
    bb = np.linalg.solve(A_.T@A_ + lam*np.eye(192), A_.T@(tt*sw))
    return bb, (Xf@bb - tt)
grpd4 = np.repeat(np.select([iI&~iN, iI&iN, ~iI&iN], ['ins','combo','ne'], 'none'), 4)
for nm_, t_ in [('rateI', tI), ('rateN', tN), ('presI', np.repeat(iI.astype(float),4)), ('presN', np.repeat(iN.astype(float),4))]:
    bb, e_ = fitb(t_)
    ed = e_[:len(Fd)]
    s_ = ' '.join(['%s max %.3f rms %.4f' % (g, np.abs(ed[grpd4==g]).max(), np.sqrt((ed[grpd4==g]**2).mean())) for g in ['ins','combo','ne']])
    print(nm_, '|b| %.0f' % np.linalg.norm(bb), s_, '| meas max %.4f' % np.abs(e_[len(Fd):]).max())
    globals()['b_'+nm_] = bb
```