Adding z noise barely changes the decoders. With ridge alpha 1:

| target | R² | slope on truth |
|---|---|---|
| insulin surprise, lag 1 | 0.33 | 0.32 |
| insulin surprise, lag 2 | 0.18 | 0.19 |
| NE deviation, lag 1 | 0.15 | 0.15 |
| NE surprise, lag 1 | 0.10 | 0.10 |

So the ridge fit stays away from the z-driven directions, and I can trust these as offline estimates of decoder quality.

Next I build the injection direction: a vector u in the head's hidden pre-activation space. For each drug, u should move its target family's predicted mean by 1 unit per unit of injected signal and leave the other family unchanged. I fit it by least squares on the exact per-reading gradients. Then I check, with the exact nonlinear head, how linear the response is for shifts up to ±15 and how much it leaks into the other family.

I am also updating the notebook now, since a restart could come soon.

```python
with open('/work/notes.md','a') as f:
    f.write("""
## Restart 6 (runs 150-165)
- Offline head exact (pred0/nll0 reproduced). WC0 = with_code at z=prior mean. wc_from(z) rebuilt (z_head, query_norm, type_emb[1], reg). One z draw changes wc rms .23.
- Factual residual slopes on insulin surprise (PT_tr eps, lags1-6, gluc rows pooled): -1.26,-1.29,-0.42,+0.37,+0.27,+0.61 (se .3) -> model misses lag1-2 effect, later lags carried by readings.
- NE: E[rate|MAP hist] via HGB (R2 .77), D_ne = rate - E. Resid slopes on D_ne lags1-3 (arterial): 3.14, 0.15, -2.2. One-step eps_ne lags: 3.37,2.65,3.15.
- Decoders from wc with z-noise aug (ridge a=1, CV): ins eps lag1 R2 .33 (slope .32), lag2 .18; NE D lag1 .15, eps_ne lag1 .10. Full state ins lag1 .63 (head cannot see slots separately).
- Plan now: hidden-space injection u (W1 += u w^T) gated by Jacobian LS so glucose-only / MAP-only; scale c/kappa; test with dose_edit both on sel1.
""")
Jh = G*gp
def solve_u(on, off, lam=10.0, rho=1e-3):
    Ja, Jb = Jh[on], Jh[off]
    Amat = Ja.T@Ja/len(on) + lam*Jb.T@Jb/len(off) + rho*np.eye(192)
    return np.linalg.solve(Amat, Ja.T@np.ones(len(on))/len(on))
u_ins = solve_u(rg, rm); u_ne = solve_u(rm, rg)
def mean_from_A(Ai, idx):
    lg = gelu(Ai)@W2.T + b2 + HI['trans'][idx]; lg -= lg.max(1,keepdims=True)
    p = np.exp(lg); p/=p.sum(1,keepdims=True)
    return (p[:,1:]*cen[idx]).sum(1)
for nm,u,on,off in [('ins',u_ins,rg,rm),('ne',u_ne,rm,rg)]:
    out=[]
    for s in (-15,-5,5,15):
        don = mean_from_A(A[on]+s*u, on)-mA[on]; dof = mean_from_A(A[off]+s*u, off)-mA[off]
        out.append((s, round(float(don.mean()),2), round(float(don.std()),2), round(float(dof.mean()),2), round(float(dof.std()),2)))
    print(nm, round(float(np.linalg.norm(u)),2), out)
```