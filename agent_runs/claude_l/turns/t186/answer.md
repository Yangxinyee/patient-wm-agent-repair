The original model's rollout response to one extra unit of insulin at hour 12 is −0.65 mg/dL at hour 13. It then stays at about −0.37 through hour 24 with no decay, so the model treats any glucose shift as permanent. The records say insulin's effect is transient: −2.06, −2.45, −2.18, … per U/h, back near 0 by lag 12. Combined with the hard prelude attention and the fast forgetting of h, this sets my plan.

Stage 1, a fallback: repurpose the dead z channel. z_post computes the dose residual from ctx and u. The z token is made a copy of u (so attention splits evenly and nothing else changes) plus a residual direction. That amounts to an amplified residual dose seen at drug bundles, with the amplification calibrated against the true readouts.

Stage 2, if feasible: add multi-hour memory so the response has the right transient shape.

Before building, I'll save these findings to the notes. I'll also check whether u is low-dimensional enough to copy through 32 z dims: the PCA of u over a grid of insulin and NE rates, including the edited out-of-range doses (x0 to x3).

```python
with open('/work/notes.md','a') as f:
    f.write("""
## Mechanism findings (runs 171-175)
- Teacher-forced dose response of ORIGINAL model: insulin +1U/h -> finger lag1 -0.62, lags>=2 ~0. All via u TOKEN; z channel DEAD
  (prelude attention gives z token ~0 weight; z_in.weight nonzero but unattended). |dz| from dose x2 ~2.3 (z_post sees u).
- Prelude attention is HARD: drug bundles: heads0-2 -> u token (w~1), head3 -> slots. reading bundles: heads1,2 -> e, head3 -> u_empty, head0 -> slots.
- h perturbation after drug bundle: glucose effect lag1 2.2, lag2 14%, lag4 1% => state forgets within ~1h (M-driven recompute).
- Rollout single-dose IRF (ins x2 at first record in [12,13)), per U/h: -0.65 then -0.37 FLAT to hour 24 (model treats shifts as permanent).
  True open-loop IRF transient (-2.06,-2.45,-2.18,-1.77,-1.29,... ~0 by lag 12).
- PLAN Stage1: repurpose dead z: z_post computes PCA-copy of u + residual r=(dose - pihat(ctx)); z_in maps to token = u + r*w
  (attention splits 50/50 between identical tokens -> no disruption; r*w acts as amplified residual dose). Calibrate gain on true readout.
  Stage2 (if feasible): multi-lag memory for transient shape.
""")
# PCA of u over a dose grid (insulin / NE presence and rates incl. OOD)
def mk_drug_atom(code, rate):
    nm=DRUGN[code]; dose = rate if code==8 else 4.8*rate
    ref=next(a for s in ks[:5] for (t,al) in BUN[s] for a in al if a['code']==code and a['kind']==0)
    return edit_atom(ref, 1.0) if False else dict(ref, num=(0,0,zsc(nm,'dose',dose),1,zsc(nm,'rate',rate),1), db=fbin_f32(nm,'dose',dose), rb=fbin_f32(nm,'rate',rate))
ins_r=[None]+list(np.arange(0,16.01,0.25)); ne_r=[None]+list(np.arange(0,3.01,0.02))
U=[]; lab=[]
for ir in ins_r:
    for nr in ne_r:
        al=[]
        if ir is not None: al.append(mk_drug_atom(8,ir))
        if nr is not None: al.append(mk_drug_atom(9,nr))
        U.append((set_enc(W0,[fix_atom2(a) for a in al],'encode_action') if al else u_empty).numpy()); lab.append((ir,nr))
U=np.array(U); mu_=U.mean(0); Uc=U-mu_
sv=np.linalg.svd(Uc,compute_uv=False); ev=sv**2/np.sum(sv**2)
print(U.shape, 'cum var', [round(float(1-ev[:n].sum()),8) for n in [4,8,16,24,28,31,32]])
```