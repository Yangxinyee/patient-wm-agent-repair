The SNR test is decisive, and negative for the z channel. At glucose and MAP readings, the best possible linear readout of the fwd8 writes has an SNR of only about 1.2 per unit² for norepinephrine at lag 1. For insulin at any lag and norepinephrine at lag 2 or more it is at most about 0.1.

A useful readout would need an SNR of roughly 60 or more for norepinephrine, given an innovation SD of 0.13, and several for insulin. So the dose signal written into z18 and z28 is buried in q's natural variation. Scaling the writes up would saturate the LayerNorm on the z token long before reaching that.

The mechanism follows from the architecture. Every new bundle enters the large raw slot memory h only as a small increment through the prelude. The heads see only LayerNorm'd slot means. So a write is only visible if it lands in directions where the state naturally varies little.

So the problem is not the writer's magnitude but its direction. A dose signal could be read cleanly only if it is placed in quiet directions of h/out/q that persist across steps. Designing that needs the step function itself: how the z or u token maps to slot changes through the prelude, blocks and coda. Then I could aim the write at quiet, persistent directions and check it offline instead of spending forward passes.

Before committing to rebuilding the step, I need to gauge the cost. This run:
- lists the remaining core parameters: blocks, age, flow, the encoders, the config's head count;
- computes the natural variance spectrum of out and q, to see whether truly quiet directions exist at all.

I will also log the negative result in the notebook.

```python
print('config', VOCAB.get('config'))
rest_=[n for n in NAMES if n.startswith('core.') and not any(s in n for s in ['prelude','coda','b_in','c_out','out_norm','delta_in','a_in','slots'])]
for n in rest_: print(n, SHAPES[n])
print([ (n,SHAPES[n]) for n in NAMES if n.startswith('encode.') or n.startswith('encode_action.')][:30])
Xall=np.r_[S2['x'],S5['x'],S6['x'],S7['x']].astype(np.float64)
outA=Xall[:,:192]; hmA=Xall[:,192:960].reshape(-1,4,192).mean(1)
for nm,A_ in [('out',outA),('mean_slots(h)',hmA),('q(gluc+map,99 stays)',Q99)]:
    ev=np.linalg.eigvalsh(np.cov(A_.T)); ev=np.sort(ev)
    print(nm,'eig smallest',np.round(ev[:6],6),' median %.4g  largest %.4g'%(np.median(ev),ev[-1]))
sdh=hmA.std(0); print('mean_slots(h) per-dim sd: min %.3g  10pct %.3g  median %.3g  max %.3g'%(sdh.min(),np.percentile(sdh,10),np.median(sdh),sdh.max()))
print('mean_slots(h) per-dim |mean| median %.3g max %.3g'%(np.median(np.abs(hmA.mean(0))),np.abs(hmA.mean(0)).max()))
with open('/work/notes.md','a') as f:
    f.write("- run 189-190: insulin rate linearly decodable from ctx at drug bundle (cvR2 .71, resid .65 vs policy .62). NE: ctx R2 .59 vs policy-features R2 .87 (innov sd .128); prev NE rate in ctx only R2 .62.\n"
            "- run 191-198: fwd8 z18/z28 writes barely reach heads: dctx .04, dout .03 (natural 1.55); dh ~1.4 per slot common (natural 76). GLS SNR of dq on lagged doses per unit^2: NE lag1 ~1.2 (gluc rows), .5 (map); insulin ~.01. => z-channel as built is USELESS (signal buried). view_from(prev,dt)[0] = q exactly; R.k/R.dt index fwd2/fwd8 rows.\n"
            "- every drug bundle has START atoms for both drugs (rate 0 when off); dose edit keeps atoms, scales rate&dose (x0 -> rate 0).\n")
```