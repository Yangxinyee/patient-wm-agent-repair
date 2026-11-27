Stages 1–3 work end-to-end in the local core:
- The change in h along the insulin direction is about 10× the decoded insulin rate (decoded 1.14 → 11.7, 1.03 → 10.4).
- The norepinephrine direction behaves the same way (decoded 0.85 → 8.5).
- Off-target change is ≤0.45, and no-action bundles show only small decayed carry-over (≤0.4).

The remaining stage is the value head: gated dead-unit pairs that read the injected q-change and tilt the glucose (or MAP) logits. First I record progress in the notebook. Then I inspect what `GRa` holds and the glucose/MAP code ids and bins, so I can build with_code samples for choosing the read direction, the code gate, and the dead units.

```python
with open('/work/notes.md','a') as fh: fh.write("""
- Runs 249-268 BUILD (session 12): free value dims NF5 = 5 lowest-energy head2 value dirs (energy 1e-4). Reads RW2 (193x5: gate w_g2 [clean, incl k=0 & mixture negs], ins r/sqrt, ne r/sqrt).
  KVn/Won: V2'=V2+NF5(W5^T-NF5^T V2); Wo2'=Wo2+(D1-Wo2 NF5)NF5^T, D1=amp*sig0*U3[:,1..5]/g3 (n3 low-var dirs), amp=(.05,.03x4).
  Prelude FFN: 4 low-impact units UNITS[:4] = gated linear-decode PAIRS (lam~24, B=13, kap=10/Lmax) -> write DOUT (E low-var dirs QE[:,1],[:,2], GE=.05).
  Linear decode limits: ins saturates ~4.9 (bias -1.8 at r>=5), ins noise at ne-only bundles sd .8; ne bias -1 at r=2.
  b_in rank-1 per drug (Bin2): reads E-channel (vEn, intercept via tvec), writes a_d=c_out^-1(10*TY)/softplus(delta_in). TY ins=a_y, ne=QY[:,1] orth.
  Local check (step_at, Wt edits via apply_local): dy.TY = 10.0 x decoded rate, off-target<.45, no-action ~0. EDIT dict holds all core edits (numpy).
  NEXT: value head dead-unit pairs gated by code (gf / map), read dq along injection, tilt logits; then save_edit + dose_edit both.
""")
print(GRa.columns.tolist()); print(GRa.head(2).to_string()[:600])
print('codes', {k:v for k,v in CODE.items() if k in ('glucose_finger','map_arterial')}, 'ncodes',len(CODE))
gfn=[d for d in VOCAB['numeric'] if d['code']=='glucose_finger'][:2]; print({k:(v if not isinstance(v,list) else np.round(v[:20],1).tolist()) for k,v in gfn[0].items()})
```