The head-level correction is cheap or even helpful for insulin. At c = −3 the predicted cost is −0.0002 nats with the lag-1 decoder and +0.0002 with the level decoder. For NE it is driven by decoder noise: A = 0.035 is solid, while B (−0.010 ± 0.027) is not distinguishable from zero on 100 stays. The predicted cost is about +0.0018 at c = 3 and +0.0047 at c = 5. Under the expected positive B from the true innovation effect, these become about +0.001 and +0.003.

Plan:
- Insulin and NE corrections both at the value head, through two repurposed hidden units gated by code.
- Compensation by last-reading bin through embed_transition.
- The NE gain is limited by the factual budget if needed.
- Calibration with dose_edit, then a factual check with 'both'.

Before building, I record these findings in the notebook; the conversation may restart soon. In the same run I check two practical things:
- which value-head hidden units are dead or least important, to repurpose;
- the code-embedding tensors and value-bin centres needed for gating and for the logit tilt.

```python
notes = """
## Session 5 (runs 141-165)
- Policy check on setA tool residuals (F2): finger res ~ insulin innov lags [-3.13,-1.63,.6] (se .63) but ~ policy part lags [-1.21,-.41,.96]
  -> policy part ABSORBED by model; raw dose amplification would bias factual (~2.8 mg/dL rms). NE noisy (similar conclusion).
- Correction must respond to dose LEVEL (sustained x2) with compensation depending on READINGS only (not past doses), else NE x2 cancels.
- u-token injection via prelude head 2 (attends 100% to u on all bundles; decoders w_ins/w_ne on norm1(u) R2 .999, zero on empty u):
  model's own dose direction useless (ins->glucose +0.38 wrong sign; NE->glucose -2.7 cross). Consistent per-query effect needs |alpha|~1000 per
  1 mg/dL/(u/h) (LN after prelude) -> NOT viable. Relay (inject to make dose decodable at head) fails: heterogeneity ~9x slope.
- HEAD-LEVEL correction c*(decoded dose - mean of decoded dose given (code,lastbin)) evaluated on setA with tool residuals:
  insulin (finger) ridge decoders from samp_pert: lag1 slope .86 errsd .57, c=-3 -> dNLL(weighted) -0.0002 (gain); level3 slope 1.0 errsd .34.
  NE (art): errsd .19-.21 (innov sd only .079); cost ~ +0.0018 at c=3, +0.0047 at c=5 (B not significant, A=.035 solid). Budget 0.005 total.
- setA reading mix: finger 34%, lab 10%, art 26%, cuff 29%; res sd 14.9/12.0/5.2/7.9.
- kernel: head_mean returns (mu, P); globals b1,b2 were stale -> reloaded W1,b1,W2,b2,TR from original. RU0 = correct rate map.
  run_rows(stays)/run_rows_wc use gen_samples(G3[s]); H100 = 100 train stays; sweep files /work/sweep_h2.npz, sweep_dwc.npy.
- PLAN: repurpose 2 value_bin hidden units (gated by code group), tilt logits along bin centres, comp via embed_transition per (code,lastbin);
  calibrate gains with dose_edit on setA vs truth (-14.17/+7.09/+4.82/-2.41); check factual with 'both'.
"""
open('/work/notes.md','a').write(notes)
print([ (n, SHAPES[n]) for n in NAMES if ('code' in n or 'type' in n or 'numeric' in n or n.startswith('embed.bins')) ][:14])
print({k: (np.round(VOCAB['value_bin_centres'][k][:3],1).tolist() if isinstance(VOCAB['value_bin_centres'], dict) else '') for k in (list(VOCAB['value_bin_centres'].keys()) if isinstance(VOCAB['value_bin_centres'], dict) else [])})
pre = XA_ @ W1.T + b1; act = gelu(pre) if 'gelu' in globals() else pre*0.5*(1+erf(pre/np.sqrt(2)))
contrib = np.abs(act).mean(0) * np.linalg.norm(W2, axis=0)
order = np.argsort(contrib); print('least-important units', order[:6].tolist(), np.round(contrib[order[:6]],4).tolist(), '| max act of those', np.round(np.abs(act[:, order[:6]]).max(0),3).tolist(), '| median contrib %.3f' % np.median(contrib))
```