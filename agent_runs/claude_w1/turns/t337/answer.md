The architecture excerpt confirms the dose signal reaches the value head only through q. q is the LayerNorm of view plus type_emb, and view includes z_head(z) with z drawn from the prior four times at evaluation. The register fields (last value, age, last rate bin, status) belong to the same reading code, so they carry no drug information.

The v4 failure therefore has two candidate causes:
- z-prior-draw noise along the huge-norm decoder rows;
- my replica's head inputs (Xf) differing systematically from the tool's.

Which one it is decides whether a redesign can work. If z draws are the cause, I can simulate them in the replica, build a decoder that is robust to them and check it on the tool. If the replica's core is off, head-level decoding is unreliable and I should fall back to a much more conservative edit.

To test this I need my replica's internals: how q and the context are computed, and how z enters. So I'll print the remaining helper names and short source excerpts of the key functions. The output is kept short.

```python
import inspect
fn=sorted(k for k,v in globals().items() if isinstance(v,types.FunctionType) and not k.startswith('_'))
print(' '.join(f for f in fn if f>'q_row')[:500])
for f,nc in [('q_of',600),('q_row',600),('ctx_of',500),('core_step',700),('gen_samples',600),('eval_det',300),('head_P',300),('init_ctx',300)]:
    if f in globals():
        try:
            src=inspect.getsource(globals()[f]).splitlines(); s=' | '.join(l.strip() for l in src[:16])
        except Exception as e: s='nosrc '+str(inspect.signature(globals()[f]))
        print('##',f, s[:nc])
```