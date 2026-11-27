The pa columns are: insulin rate, norepinephrine rate, insulin present, norepinephrine present.

A read that must also be ~0 on all non-dose tokens gives these residuals on dose tokens:

| Target | Residual sd | Target sd |
|---|---|---|
| Insulin present | 0.052 | — |
| Insulin rate | 0.49 | 1.15 |
| Norepinephrine rate | 0.25 | 0.33 |
| Norepinephrine present | 0.20 | 0.24 |

So insulin presence is clean, but the rates cannot be read cleanly while staying zero on the other tokens. Unconstrained, the rate reads are clean on dose tokens: R² .995 and .98 in earlier sessions.

**Conclusion:** split the job.
1. A clean "dose token present" gate, read from head2.
2. A rate read that is accurate on dose tokens, with only mild limits on the other tokens.
3. A GELU unit pair in the prelude FFN, which has 154 dead units. It gives rate × gate on the slot tokens before anything enters the persistent state, so leakage on no-dose bundles is cut off by the gate.
4. The deviation-form compensation c·m(band) goes at the value head, gated on the gf code and on the last gf reading being fresh (age about 1 h). That uses only head inputs.

**This step:** check the stored prelude arrays (n1, patt). Then fit three reads: "any dose token" presence, and the insulin and norepi rates at several strengths of the limit on non-dose tokens.

**Test:** if "any dose" presence reaches residual < 0.05 with near-zero values on non-dose tokens, and a rate read reaches residual < 0.1 on dose tokens with non-dose spread of a few units at most, the gated prelude stage is viable.

```python
for nm in ['n1','patt','act','te','mu','evec','ins_s','ne_s','SSTp','TOKS','H_','HOc']:
    v=globals().get(nm); print(nm, type(v).__name__, getattr(v,'shape',None) if not isinstance(v,list) else len(v))
Xu=U_act; Xn=np.vstack([NEG, U0[:1]])   # include u0 token (= norm1 bias)
c_any,sp_,mn,sn,mx=ridge_read(Xu,np.ones(len(Xu)),Xn,lam_n=1.0)
print('ANY: act resid %.3f | NEG mean|.| %.4f sd %.4f max %.3f | u0 val %.3f'%(sp_,mn,sn,mx,U0[0]@c_any[:-1]+c_any[-1]))
for j,nmj in [(0,'ins'),(1,'ne')]:
    for ln in [1e-4,1e-3,1e-2,1e-1]:
        c,sp_,mn,sn,mx=ridge_read(Xu,pa[:,j],Xn,lam_n=ln)
        print(nmj,'lam_n %.0e: act resid %.3f | NEG sd %.3f max %.2f'%(ln,sp_,sn,mx))
```