Lab glucose is missing completely at random: it is present about 19% of the time regardless of insulin or glucose level. That makes it an unbiased outcome for long-lag effects. Finger-sticks are not, because they are taken hourly when insulin is running.

For MAP, about half the stays have no arterial line. Arterial readings are more frequent when norepinephrine is running (63% vs 35%) and cuff readings less frequent. So MAP has its own selection, to deal with later.

Now the deconvolution for insulin. I'll estimate:
- T_k: the effect of the random rate at h on lab glucose at h+k, for k = 1..12;
- D_m: its effect on later recorded rates.

Then I'll solve recursively for the direct kernel K_k, with band fixed effects and the current/previous reading as controls. For comparison I'll repeat it with finger glucose as the outcome, which is subject to selection.

```python
Pp = P.copy()
L = 13
g = Pp.groupby(level='stay_id')
for k in range(1,L):
    Pp[f'gl_l{k}'] = g.gl.shift(-k); Pp[f'gf_l{k}'] = g.gf.shift(-k); Pp[f'd_l{k}'] = g.ins0.shift(-k)
Pp['gdec'] = Pp.gf.combine_first(Pp.gl)
Pp['g_prev'] = g.g_last.shift(1)
S = Pp[(Pp.index.get_level_values('h')>=1) & Pp['ins'].notna() & Pp.gdec.notna() & Pp.g_prev.notna()].copy()
S['band'] = np.select([S.gdec<180, S.gdec<220, S.gdec<250], [1,2,3], default=4)
S = S[S.band<=3]
def eff(s, outc):
    X = [s.gdec, s.g_prev, s.ins_prev, (s.band==2).astype(float), (s.band==3).astype(float), s.gdec*(s.band==2), s.gdec*(s.band==3), s['ins']]
    res, n, sd = ols(s[outc], X, ['gdec','gprev','insprev','b2','b3','gb2','gb3','rate'])
    return res.loc['rate','b'], res.loc['rate','se'], n
T = {}; Tf = {}; D = {0:1.0}
for k in range(1,L):
    T[k] = eff(S, f'gl_l{k}'); Tf[k] = eff(S, f'gf_l{k}'); D[k] = eff(S, f'd_l{k}')[0]
def deconv(T, D, L):
    K = {}
    for k in range(1,L):
        K[k] = (T[k] - sum(K[k-m]*D[m] for m in range(1,k)))/D[0]
    return K
Kl = deconv({k:T[k][0] for k in T}, D, L); Kf = deconv({k:Tf[k][0] for k in Tf}, D, L)
print(" k   T_gl(se)        T_gf(se)       D_k     K_gl    K_gf")
for k in range(1,L):
    print(f"{k:2d}  {T[k][0]:6.2f}({T[k][1]:.2f})  {Tf[k][0]:6.2f}({Tf[k][1]:.2f})  {D[k]:6.3f}  {Kl[k]:6.2f}  {Kf[k]:6.2f}")
```