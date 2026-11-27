The structural norepinephrine response can now be recovered. After the band × previous-rate interactions, the placebo is acceptable (+0.12 to +0.19, about 1–1.7 SE). The sqrt warnings are only from collinear empty cells.

The reduced-form MAP response peaks at about +5.8 mmHg at 3 h and decays slowly, reaching +1.6 at 12 h. An innovation at t persists in later rates: 0.77 at t+1, 0.61 at t+2, 0.49 at t+3, 0.12 at t+12. Solving the triangular system by hand gives b1 ≈ 3.37, b2 ≈ 2.56, b3 ≈ 1.80, b4 ≈ 0.98 mmHg per mcg/kg/min. So norepinephrine acts quickly and wears off over a few hours.

Next I will:
1. Code the triangular deconvolution for both drugs, using the insulin propagation the same way.
2. Check NE linearity by estimating the 1-h effect within previous-rate strata (0.2–0.4, 0.4–0.7, 0.7–1.0).

Linearity matters because ×2 pushes rates beyond 1.0. A similar per-unit effect across strata supports linear extrapolation; a shrinking effect at high rates would indicate saturation.

```python
def deconv(rf_y, rf_d):
    # rf_y[L-1] for L=1..K ; rf_d[k] for k=0..K-1 (rf_d[0]=1)
    K = len(rf_y); b = np.zeros(K)
    for L in range(1,K+1):
        s = sum(b[j-1]*rf_d[L-j] for j in range(1,L))
        b[L-1] = (rf_y[L-1] - s)/rf_d[0]
    return b
ne_rfy = ne_tab[ne_tab.L>=1].map_b.values
ne_rfd = np.r_[1.0, ne_tab[ne_tab.L>=1].ne_b.values[:-1]]
b_ne = deconv(ne_rfy, ne_rfd)
print("NE structural IRF:", b_ne.round(3), "cum:", np.cumsum(b_ne).round(2))
# insulin propagation: effect of D_s noise on D_{s+k} (fill 0)
def ins_rf(L, outcome):
    Pg = P.groupby(level=0)
    hh = P.index.get_level_values('h').values
    if outcome=='Dfill':
        Y = (Pg.D.shift(-L)).where(hh+L<=24)
    else:
        Y = Pg[outcome].shift(-L)
    q = P.assign(Y=Y)
    q = q[q.glucose_finger.notna() & (q.tf>=1) & q.insulin.notna() & q.Y.notna() & (q.index.get_level_values('h')>=0)]
    q = q[~q.glucose_finger.round().isin([150,180,220,260])]
    g = q.glucose_finger.values
    X = [np.ones(len(q)), g, g**2/100, g**3/1e4] + [(q.tf.values==t).astype(float) for t in [2,3,4]] + [q.D.values]
    r = ols(q.Y.values, X, ['c','g','g2','g3','t2','t3','t4','D']).loc['D']
    return r.b, r.se
ins_rows = [(L,)+ins_rf(L,'glucose_finger')+ins_rf(L,'Dfill') for L in range(1,13)]
ins_tab = pd.DataFrame(ins_rows, columns=['L','g_b','g_se','d_b','d_se'])
print(ins_tab.round(4))
b_ins = deconv(ins_tab.g_b.values, np.r_[1.0, ins_tab.d_b.values[:-1]])
print("Insulin structural IRF:", b_ins.round(3), "cum:", np.cumsum(b_ins).round(2))
# NE linearity: 1h effect by NEp strata
for lo,hi in [(0.05,0.2),(0.2,0.4),(0.4,0.7),(0.7,1.0)]:
    Y = P.groupby(level=0).map_arterial.shift(-1)
    q = P.assign(Y=Y); q = q[(q.index.get_level_values('h')>=1)&q.NEp.notna()&q.NE.notna()&q.Y.notna()&q.mr.notna()&(q.NEp>=lo)&(q.NEp<hi)]
    X = build_ctrl_ne(q)+[q.NE.values]
    r = ols(q.Y.values, X, [f'x{i}' for i in range(len(X)-1)]+['NE']).loc['NE']
    print("NEp",lo,hi,len(q), round(r.b,3), round(r.se,3), " mean NE", q.NE.mean().round(3))
```