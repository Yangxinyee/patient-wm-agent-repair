The edit behaves exactly as designed on the real drug-bundle inputs:
- the change in z mean matches the intended injection to 1e-5;
- the four repurposed units contributed nothing (1e-5) at drug bundles before;
- the gates sit at ≥+24.7 when on and ≤−43.8 when off.

I now save the edit (κ_I = 2, κ_N = 4) and record the design in the notebook. The next step is a dose_edit pass to measure the model's actual response gain and time profile. The CPU replica cannot show how a z shift at a drug bundle propagates through the core.

```python
save_edit('z_post.0.weight', W0n.float()); save_edit('z_post.0.bias', b0n.float()); save_edit('z_post.2.weight', W2n.float())
print('edited:', edited())
with open('/work/notes.md','a') as f_:
    f_.write("""
## Runs 85-100: INJECTION EDIT v1 (z_post only)
- u_k (encode_action output) loses insulin magnitude via LayerNorm (sigma(y) 17..119), but a ridge readout from u IS exact for
  insulin-only (rms .0002, rates<=11.5; z clip at 10 => rates>11.5 indistinguishable). Joint readouts fI (I-only+combos), fN:
  mean error of readout change over tuning drug bundles ~1% (ins, all m), NE <1% (m=.5,2), 5% m=0, -10% m=3.
- Gates: LinearSVC on u: insulin present (on>=.886, off<=-.355), NE present C=1000 (on>=.736, off<=-.001). thresholds = midpoints.
- E[rate|ctx]: ridge on CTX3 at drug-present bundles (fwd_3), CV R2 ins .56, NE .31 (model's own knowledge).
- z direction: regress z post mean on reading surprises: finger==lab direction (cos 1.0), art==cuff (cos 1.0); per-unit norms
  finger .0233/mgdl, lab .0094, art .069/mmHg, cuff .045. Use D4[:,0] (finger) for glucose, D4[:,2] (art) for MAP.
- z_post units repurposed (low worst-case UB, dead at drug bundles): %s = [A_ins, gate_ins, B_ne, gate_ne].
  unit pairs GELU(P+x)-GELU(P), P=M*(gate-th), x=c*(f(u)-E(ctx)); dz = -kapI*s_ins*D4[:,0] + kapN*s_ne*D4[:,2].
- v1 saved: kapI=2.0, kapN=4.0 (cI=.01,cN=.05,MI=40,MN=60). Replica check at drug bundles exact (1e-5).
""" % (units4,))
```