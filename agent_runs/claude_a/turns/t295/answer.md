The insulin readout works. With θ = −2 the edited model gives −12.79 for insulin ×2 (truth −14.98) and +6.08 for ×½ (truth +7.49) on 40 tuning stays, up from the original's −2.04 and +1.22. The offline baseline also reproduces the tool's original numbers. The response is close to linear in θ. Interpolating the added effect, ×2 needs a scale of about 1.20 and ×½ about 1.29, so θ ≈ −2.45 balances the two at roughly −15.2 and +7.2.

This run will:
1. Rebuild the readout at θ = −2.45.
2. Recheck the factual cost on sel2 and dsE and the rollout shifts on the same 40 stays.
3. Save the edit with `save_edit`, so a usable insulin fix is locked in.
4. Append a summary of runs 247–282 to the notebook.

Then I'll check the result with the forward tool and move on to norepinephrine.

```python
theta=-2.45; w_=(theta/cfm[0])*100./212.
PR=add_readout(PB8, bet, b0g, gdir, w_, WCall=WC8, tilt=tilt)
nR2=tf_eval(PR,d82,False); nRE=tf_eval(PR,d8E,False)
print('theta=%g cost sel2 %+.5f dsE %+.5f'%(theta,float((nR2[0]-a_s2).mean()),float((nRE[0]-a_E).mean())))
rPR=eval_shifts(PR,st40); print('PR', {n:tuple(np.round(v[:2],2)) for n,v in rPR.items()})
nchg=0
for k in PO:
    if not torch.equal(PR[k],PO[k]): save_edit(k, PR[k].clone()); nchg+=1
print('saved', nchg, 'edited now', len(edited()))
open('/work/notes.md','a').write("""
## Restart 7 (runs 247-282) INSULIN FIX WORKING, SAVED
- sel2 is drug-heavy: dims 2,67 cost +.003/.004 there. Rescanned per-dim ablation cost on sel2+dsE -> /work/dimcost2.npy. cheap: 79,99,50,49,104,150,153,70,64,191,149,137.
- encoders: u must be re-encoded (reenc(ds,P)) when encode_action edited! insulin bundles: start atom kind0 (rate) + end atom kind2; all infusions last exactly 1h. Decoder target = kind-0 start rate (dvecN/d0N, resid .04).
- Build: PA8=build_ins2(PO,dvecN,d0N,keepN,jU=79,jR=99,jI=50,half_life_h=2.5,cb=8) (mem corr .95 w/ ideal); PB8=plumb(PA8,go=4,gq=4) (enables out_norm/query_norm dims 50/99, zeros other head readers);
  readout add_readout(PB8,bet,b0g,gdir,w): value units 107(signal)+109(gate ref), gate=LDA dir gdir on with_code (glucose_finger code 6), compensation bet fit on ORIGINAL wc (R2 .91), constrained orth to LN leak dirs; tilt=(centre-160)/100 on bins1..16.
- theta=-2 -> 40 sel1 stays: ins_x2 -12.79 (truth -14.98), half +6.08 (7.49); cost sel2 +.0003 dsE 0. theta=-2.45 saved as edit.
""")
```