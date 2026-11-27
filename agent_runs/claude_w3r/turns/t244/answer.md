The findings settle the design.
- The z-scores are exactly linear in the rate: insulin z = r − 1.5; norepinephrine z = (r − 0.46)/0.6.
- But `encode_action.mlp1` has no dead units, and the pre-norm scale of the encoder output varies a lot (cv 0.92). Building a clean channel inside the encoder would get scrambled by its final LayerNorm, so I drop that route.
- The normalised prelude tokens live in a low-dimensional subspace: 66 of 192 directions have variance below 1e-4.

**Conclusion:** use route (α), the gated prelude channel.
- One replaced head2 value dimension gives a clean gate.
- Another gives a linear rate read.
- A GELU unit pair in the prelude FFN multiplies rate by gate on the slot tokens.

**Two refinements:**
1. Gate insulin on *insulin present* rather than *any dose*. Insulin presence reads cleanly (resid 0.052). The insulin read looked wrong on norepinephrine-only tokens, and this gates them off.
2. Fit each rate read on factual tokens plus synthetic ×2 and ×½ tokens made with `u_of`, so its response to the evaluation's edits is as linear as possible.

**This run:**
1. Validate `u_of` on all 2406 dose bundles, including one-drug bundles.
2. Fit the gate and the two augmented rate reads.
3. Report residuals for the factual, ×2 and ×½ groups, and the edit-response slope (ideal 1).
4. Record runs 226–230 in the notebook.

**Test:** slopes near 1 and residuals ≲0.15 would confirm. A slope far from 1 under ×2 would mean the read needs a nonlinear stage.

```python
ins_p=pa[:,2]>0.5; ne_p=pa[:,3]>0.5
n1u=lambda a,b: n1_of(np.asarray(u_of(float(a),float(b))).reshape(-1))
err=max(np.abs(n1u(pa[i,0]*ins_p[i],pa[i,1]*ne_p[i])-U_act[i]).max() for i in range(len(pa)))
print('u_of validation max err over all 2406:', err)
iI=np.where(ins_p)[0]; iN=np.where(ne_p)[0]
XI2=np.array([n1u(2*pa[i,0],pa[i,1]*ne_p[i]) for i in iI]); XIh=np.array([n1u(.5*pa[i,0],pa[i,1]*ne_p[i]) for i in iI])
XN2=np.array([n1u(pa[i,0]*ins_p[i],2*pa[i,1]) for i in iN]); XNh=np.array([n1u(pa[i,0]*ins_p[i],.5*pa[i,1]) for i in iN])
def fitg(Xp,yp,Xn,lam_n,lam=1e-3):
    c=ridge_read(Xp,yp,Xn,lam_n=lam_n,lam=lam)[0]; return c
res={}
for ln in [1e-4,1e-3]:
    cI_=fitg(np.vstack([U_act[iI],XI2,XIh]),np.r_[pa[iI,0],2*pa[iI,0],.5*pa[iI,0]],Xn,ln)
    f0,f2,fh=rd(cI_,U_act[iI]),rd(cI_,XI2),rd(cI_,XIh); r=pa[iI,0]
    s2=np.polyfit(r,f2-f0,1)[0]; sh=np.polyfit(-.5*r,fh-f0,1)[0]
    print('INS ln %.0e: resid fact %.3f x2 %.3f half %.3f | slope x2 %.3f half %.3f | NEG sd %.2f max %.1f'%(ln,(f0-r).std(),(f2-2*r).std(),(fh-.5*r).std(),s2,sh,rd(cI_,Xn).std(),np.abs(rd(cI_,Xn)).max()))
    cN_=fitg(np.vstack([U_act[iN],XN2,XNh]),np.r_[pa[iN,1],2*pa[iN,1],.5*pa[iN,1]],Xn,ln)
    g0,g2,gh=rd(cN_,U_act[iN]),rd(cN_,XN2),rd(cN_,XNh); q=pa[iN,1]
    t2=np.polyfit(q,g2-g0,1)[0]; th=np.polyfit(-.5*q,gh-g0,1)[0]
    print(' NE ln %.0e: resid fact %.3f x2 %.3f half %.3f | slope x2 %.3f half %.3f | NEG sd %.2f max %.1f'%(ln,(g0-q).std(),(g2-2*q).std(),(gh-.5*q).std(),t2,th,rd(cN_,Xn).std(),np.abs(rd(cN_,Xn)).max()))
    res[ln]=(cI_,cN_)
cpI=fitg(np.vstack([U_act,XI2,XIh,XN2,XNh]),np.r_[ins_p*1.0,np.ones(len(XI2)+len(XIh)),ins_p[iN]*1.0,ins_p[iN]*1.0],Xn,1.0)
pv=rd(cpI,np.vstack([U_act,XI2,XIh,XN2,XNh])); tv=np.r_[ins_p*1.0,np.ones(len(XI2)+len(XIh)),ins_p[iN]*1.0,ins_p[iN]*1.0]
print('INS-PRESENCE gate: resid %.3f, min on present %.3f, max on absent %.3f, NEG max %.3f, u0 %.3f'%((pv-tv).std(),pv[tv>0.5].min(),pv[tv<0.5].max(),np.abs(rd(cpI,Xn)).max(),rd(cpI,U0[:1])[0]))
with open('/work/notes.md','a') as f: f.write("""- Runs 226-230: pa cols=(ins rate, ne rate, ins present, ne present) for U_act (n1(u) at 2406 drug bundles; u_of(ins,ne) reproduces them).
  NEG=n1 of 6 non-u tokens x 5201 bundles. ANY-dose gate read (192-dim on n1): resid .012, NEG max .07, u0 .005 -> clean.
  Rate reads with NEG constraint: ins resid .15-.18 (saturates >4), ne .14-.16 (poor). head2 value out-energy rank~2 (bottom-3 dims 1e-9) -> free value dims.
  z_rate linear (ins z=r-1.5; ne z=(r-.46)/.6). encode_action mlp1 no dead units, mlp2 128 dead; pre-norm scale cv .9 -> encoder route dropped.
  DESIGN (alpha): head2 value dims -> [ins-presence gate, ins rate read, any gate, ne rate read] -> W_o cols into low-var x_slot dirs ->
  prelude FFN dead-unit GELU pairs (gate*rate) -> E dir -> b_in rank-1 -> h_mid -> y -> head dead units (+ fresh-gf x band compensation).
""")
```