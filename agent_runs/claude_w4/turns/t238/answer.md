`step_full`'s bytecode shows how a core step actually works. This matters because it explains why the earlier edits had so little effect.

- **The prelude output E is not passed straight into the new slots.** The block adds E to the decayed previous slots, then runs self-attention, cross-attention to a memory of past events, and an FFN. The new slots are hn = decay·h + softplus-step·(b_in·E) + the block's increments. E itself is subtracted back out.
- **The prediction state is built after that.** `out` is a layer norm of the slot mean after a c_out projection and a small "coda" (attention plus FFN).
- **The prelude output is itself layer-normed.** With slot norms near 800, an injected vector is shrunk roughly 50-fold before it reaches the step, and then reshaped by b_in and the coda.

So dI was probably designed in the wrong space: possibly for E, while it was injected before the prelude norm. A correct direction needs the Jacobian of the step outputs (hn and `out`) with respect to a vector added to the head-1 output, which needs the CPU replica.

Next I'll locate the replica's inputs:
- the bytecode of `prelude`, to see what it takes and returns;
- every array in memory sized to the 2,319 drug bundles: candidate token arrays, E arrays, teacher-forced slots `htf`, and targets.

If I find raw tokens plus E and htf, and `step_full` reproduces the recorded states from them, I can compute the needed Jacobian exactly. Otherwise I'll fall back to measuring the response empirically with leak-free probe edits.

```python
c=prelude.__code__; print('prelude', c.co_varnames[:c.co_argcount], prelude.__defaults__, '\n names', c.co_names)
t=brief(prelude)
print(' ; '.join(x.split(' ',1)[-1] if x.startswith(('LOAD_FAST','LOAD_GLOBAL','LOAD_ATTR','LOAD_CONST','STORE_FAST')) else x.split(' ')[0] for x in t)[:2500])
for nm in ['step_cv','fitstat']:
    c=globals()[nm].__code__; print('\n',nm,c.co_varnames[:c.co_argcount],'names',c.co_names)
print()
for k,v in list(globals().items()):
    if k.startswith('_'): continue
    if isinstance(v,(np.ndarray,torch.Tensor)) and v.ndim>=1 and v.shape[0] in (2319,2320,520,300):
        print(f'{k}:{tuple(v.shape)}', end=' | ')
for k in ['yt','ot','Ed','E_d','Ept','htf','HT','Hd']:
    if k in globals(): print('\nHAVE',k, getattr(globals()[k],'shape',type(globals()[k])))
```