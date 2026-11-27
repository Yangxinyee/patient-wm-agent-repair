The edit is saved; only the notebook append failed, because the text contains literal percent signs that clashed with % formatting. I rewrite the note with plain concatenation first, since a restart would otherwise lose the design details, and then I will run the forward test.

```python
txt_ = """
## Runs 85-100: INJECTION EDIT v1 (z_post only)
- u_k (encode_action output) loses insulin magnitude via LayerNorm (sigma(y) 17..119), but a ridge readout from u IS exact for
  insulin-only (rms .0002, rates<=11.5; z clip at 10 => rates>11.5 indistinguishable). Joint readouts fI (I-only+combos), fN:
  mean error of readout change over tuning drug bundles ~1 pct (ins, all m), NE <1 pct (m=.5,2), 5 pct m=0, -10 pct m=3.
- Gates: LinearSVC on u: insulin present (on>=.886, off<=-.355), NE present C=1000 (on>=.736, off<=-.001). thresholds = midpoints.
- E[rate|ctx]: ridge on CTX3 at drug-present bundles (fwd_3), CV R2 ins .56, NE .31 (model's own knowledge).
- z direction: regress z post mean on reading surprises: finger==lab direction (cos 1.0), art==cuff (cos 1.0); per-unit norms
  finger .0233/mgdl, lab .0094, art .069/mmHg, cuff .045. Use D4[:,0] (finger) for glucose, D4[:,2] (art) for MAP.
- z_post units repurposed (low worst-case UB, dead at drug bundles): """ + str(units4) + """ = [A_ins, gate_ins, B_ne, gate_ne].
  unit pairs GELU(P+x)-GELU(P), P=M*(gate-th), x=c*(f(u)-E(ctx)); dz = -kapI*s_ins*D4[:,0] + kapN*s_ne*D4[:,2].
- v1 saved via build_edit(2.0, 4.0, units4) (cI=.01,cN=.05,MI=40,MN=60). Replica check at drug bundles exact (1e-5).
"""
with open('/work/notes.md','a') as f_: f_.write(txt_)
print('notes ok', len(txt_))
```