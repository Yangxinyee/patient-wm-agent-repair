Run 15 gives the full time profile of the insulin effect. The total effect of one extra unit/h at hour t is about −2.0 mg/dL at lag 1, −2.2 at lag 2, −1.8 at lag 3, −1.3 at lag 4, −0.7 at lag 5 and −0.4 at lag 6, and roughly 0 from lag 8. Finger and lab readings agree.

The feedback onto later insulin is small: −0.05 to −0.07 units per unit at lags 1–3. The probability of a finger reading falls slightly with dose (−0.01 per unit). That is a mild selection effect, and the lab readings, which don't show it, give the same answer.

So insulin has a transient, real effect lasting about 6–7 hours. Next I'll deconvolve the feedback to get the fixed-dose impulse response ψ_k. I'll also test whether the effect is linear in rate, using I² within tier 2 where rates span 1–4. Linearity decides whether ×2 and ×½ give mirror-image answers.

```python
Tk = np.array([T[k][0] for k in range(1,13)]); Ck = np.array([C[k][0] for k in range(1,13)])
Tlk = np.array([Tl[k][0] for k in range(1,13)])
def deconv(T, C):
    psi = np.zeros(len(T))
    for k in range(len(T)):
        psi[k] = T[k] - sum(C[s-1]*psi[k-s] for s in range(1,k+1))
    return psi
psi_f = deconv(Tk, Ck); psi_l = deconv(Tlk, Ck)
print('psi finger', psi_f.round(2), 'sum1-8', psi_f[:8].sum().round(2))
print('psi lab   ', psi_l.round(2), 'sum1-8', psi_l[:8].sum().round(2))
# linearity check within tier 2 and tier 1
for tier in [1,2,3]:
    sub = base_ins[base_ins.tier==tier].copy(); sub['I2'] = sub.I**2
    for k in [1,2,3]:
        b,se,n = ols(sub, f'gf_p{k}', ['glucose_finger','I','I2'])
        print(f'tier{tier} lag{k}: I {b["I"]:.2f}±{se["I"]:.2f}  I2 {b["I2"]:.3f}±{se["I2"]:.3f}')
# nonparametric: mean residual change by rate within tier 2
sub = base_ins[base_ins.tier==2].copy()
sub['d2'] = sub.gf_p2 - sub.glucose_finger
print(sub.groupby('I').d2.agg(['mean','size']).round(2).T.to_string())
```