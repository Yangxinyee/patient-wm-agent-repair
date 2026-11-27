Run 65 shows that ridge regularisation makes the selective functional better as well as tamer. With ridge 10, |v| is 2.5 and held-out leakage at reading bundles falls to sd 0.117, about 7% of z26's natural sd there. That beats the unregularised fit, which was overfitting. Fidelity at drug bundles stays almost exact (residual sd 0.01 to 0.03). The 4-op factorization is then well conditioned (condition number about 22, alphas within -0.7 to 6.2) and float32-exact.

Next I will build the experimental edit on `z_in.weight`. It reads v·z, which is about 0 at reading bundles and equals z26 minus the type mean at drug bundles, and writes that ×α along the model's own z26 column. I use α = 4, so z26 deviations at drug bundles are scaled ×5. I also add a constraint that pins the mean leakage at reading bundles to 0, and fit on all 100 stays.

The forward pass after this will show whether z26 is the channel that carries dose effects. Insulin x2 moving well past -2.5 toward -13 would confirm it. Little change would refute it.

```python
def fit_v_full(Zread, cts, mts, lam=1e4, ridge=10.0, lam_mean=1e3):
    others=[j for j in range(32) if j!=26]
    A=Zread[:,others]; y=-Zread[:,26]
    Ac=np.array([c[others] for c in cts])*np.sqrt(lam); yc=-np.array(mts)*np.sqrt(lam)
    mu=Zread.mean(0); Am=(mu[others]*np.sqrt(lam_mean*len(Zread)))[None,:]; ym=np.array([-mu[26]*np.sqrt(lam_mean*len(Zread))])
    AA=np.vstack([A,Ac,Am,np.sqrt(ridge)*np.eye(31)]); yy=np.concatenate([y,yc,ym,np.zeros(31)])
    w,_,_,_=np.linalg.lstsq(AA,yy,rcond=None)
    v=np.zeros(32); v[26]=1; v[others]=w
    return v
# held-out check with mean constraint
ve=fit_v_full(zo_[mr&ev],cts_e,mts_e)
fo=zo_[mr&~ev]@ve
print('HO with mean constraint: |v| %.2f leak mean %.4f sd %.4f'%(np.linalg.norm(ve),fo.mean(),fo.std()))
v_z26=fit_v_full(Zr,cts,mts)
fr=Zr@v_z26
print('full fit: |v| %.2f in-sample leak mean %.4f sd %.4f'%(np.linalg.norm(v_z26),fr.mean(),fr.std()))
for t in types:
    m=grp_[t]; f=zo_[m]@v_z26; dev=zo_[m,26]-zo_[m,26].mean()
    print(t,'resid sd %.4f'%((f-dev).std()))
np.save('/work/v_z26.npy',v_z26)
alpha=4.0
dops,c=decomp_balanced(e26,v_z26,alpha)
print('cond %.1f'%c,[round(a,3) for _,a in dops])
clear_ops()
for vec,al in dops:
    rank1('z_in.weight', vec, float(al), side='in')
print(ops())
with open('/work/notes.md','a') as fh:
    fh.write("""
## z26 lever (runs 61-66)
- At ALL drug bundles only z26 varies (ins 2.31, ne 2.23, both 2.18 var); other z dims ~const. Reading bundles: many dims vary, z26 var 2.48
  but z26 ~ linear fn of other z dims (R2~0.999). z26 at ins bundles ~ 1.43*ins_rate (R2 .53); at ne bundles ~ 2.41*ne_rate (R2 .38).
- Selective edit: z_in.weight <- W (I + a e26 v^T), v.z ~0 at reading bundles (ridge-10 LS, HO leak sd ~0.12 vs z26 sd 1.57),
  v.z = z26 - type mean at drug bundles. Implemented as 4 rank1 'in' ops (balanced SPD factorisation, cond ~22). v saved /work/v_z26.npy.
- E1: alpha=4 (z26 deviation x5 at drug bundles). Testing in pass 8.
""")
```