The replica code matches the architecture document, and the shapes confirm three things:
- **`a_bar` is fixed per dimension** and not input-dependent, at about 0.27–0.47 per step.
- **The coda is a residual transformer layer**, so the state is passed through a mix-and-rewrite step after every bundle.
- **No dimension of `h` is idle.** A clean long-lived memory would require taking dimensions away from the model in every layer norm. That is a large, risky surgery for the factual fit.

A cheaper route is a z-channel loop. At each bundle, `z_post` would (a) reproduce `u` so the z token shares `u`'s attention, (b) add the current dose residual, and (c) re-add a decayed copy of the residual it reads back from the context. That gives an exponentially fading dose memory without touching the core.

Before building it I need the helper functions in the kernel: bundle splitting/encoding, `u_empty`, measurement tables, and any teacher-forcing or error-regression helpers I wrote earlier. I also want to record the architecture facts in the notebook in case of another restart.

How I'll judge the result: if I already have a teacher-forced loop and a policy-residual regression, I can go straight to measuring how an injected token changes the next glucose prediction. Otherwise I write that loop next.

```python
import inspect
with open('/work/notes.md','a') as fh:
    fh.write("""
## Restart 5 (runs 177+)
- Core step (exact, wm2.bstep): hf=h*exp(-dt*exp(flow.a)); ctx=out+LN_out(mean hf)+gap; z=z_post([ctx,e,u])[:32] (MLP 576-192-64 gelu);
  prelude tokens [hf+slots(4), e, u, z_in(z)] -> pre-norm layer -> E=prelude_norm(slots); blocks(2 layers self/cross/ffn) on hf+E;
  h2=abar*hf+sp*b_in(E)+(blocks inc); y=coda(c_out h2) (residual self layer); out=LN_out(mean y). abar const per dim (.27-.47).
- z_post output used ONLY via z token (unattended) -> free to repurpose. z_prior/z_head: head-side adapter (z_head currently =0 edit, value NLL -0.0015).
- Plan: z token = copy of u (so heads0-2 split 50/50 u/z) + w*(residual dose [+ decoded memory from ctx -> z-loop memory]).
""")
for nm in ['split_fix','enc_eu2','fix_atom2','bset_enc','set_enc','corr_effect','errreg','dosemat','core','ev','polfeats','hourly','build_bundles']:
    if nm in globals() and callable(globals()[nm]):
        try:
            src=inspect.getsource(globals()[nm]); print('#####',nm,len(src)); print(src[:1800])
        except Exception as ex: print('#####',nm,'no source',ex)
    else: print('#####',nm,'MISSING')
for nm in ['MEASC','CENd','MED','u_empty','X0row','BUN','Xs','row_start','sid_tr','sid_tu']:
    v=globals().get(nm)
    if v is None: print(nm,'MISSING'); continue
    if isinstance(v,dict): print(nm,'dict',len(v),list(v.keys())[:5])
    elif hasattr(v,'shape'): print(nm,'shape',tuple(v.shape))
    else: print(nm,type(v),(len(v) if hasattr(v,'__len__') else v))
```