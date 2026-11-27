The v3 diagnosis is now complete. It has three defects that together explain both the null dose answer and the hidden damage to factual fit.

1. **Wrong directions.** dI and dN are almost orthogonal to the model's own glucose and MAP directions in the slots (cos −0.06 and +0.045). The insulin push, −dI, even nudges the glucose belief slightly upward. That matches the dose_edit answer moving the wrong way.
2. **Leaky reading.** Head 1 is not always locked on the action token. On average it gives the slots 0.2% attention, but at some drug bundles up to 72%, and the event token up to 0.05%. The readouts give values in the thousands on those tokens, so the injection fires at the wrong bundles and with the wrong size.
3. **Real factual damage.** A ridge readout from the previous bundle's `out` predicts the original model's predicted glucose mean almost perfectly (R² 0.993; MAP 0.98). Applied to the v3 states, it implies prediction shifts with sd 2.8 mg/dL on finger and up to 19 mg/dL. That would have broken the 0.005-nat budget.

The ridge result also gives a clean target. Because `out` almost fully determines the next prediction, the injection should be designed so that the change it causes in `out` (and the slots) lies along the glucose or MAP readout.

The v4 design will use:
- readouts orthogonal to every non-action token (slots, the initial state `core.slots`, event and z tokens), so nothing leaks;
- an injection direction chosen with the CPU step replica so that the resulting change lands on the prediction readout.

To use the replica correctly, I first need to know how its functions consume their arguments. In particular I need to see whether `step_full`'s `dE` adds a perturbation after the prelude attention, which would make it the Jacobian tool I need. Next I'll disassemble `step_full`, `out_from_y` and `ctx_of`, keeping only their global and attribute references and where `dE` and `pre` are used. I'll also inspect the shape of `core.slots`.

```python
import types, dis, io, contextlib
def brief(fn, keep=('LOAD_GLOBAL','LOAD_ATTR','LOAD_FAST','LOAD_CONST','CALL','BINARY_OP','STORE_FAST','POP_JUMP','COMPARE_OP','RETURN')):
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf): dis.dis(fn)
    toks=[]
    for l in buf.getvalue().splitlines():
        p=l.split()
        if not p: continue
        for op in keep:
            if op in l:
                i=[j for j,x in enumerate(p) if x.startswith(op)]
                if i: toks.append(' '.join(p[i[0]:])[:46].replace('(NULL|self + ','.').replace('(NULL + ','').replace(')',''))
                break
    return toks
for fnm in ['step_full','out_from_y','ctx_of']:
    fn=globals()[fnm]; c=fn.__code__
    print(f'== {fnm}{c.co_varnames[:c.co_argcount]} defaults={fn.__defaults__}\n names={c.co_names}')
    t=brief(fn)
    s=' ; '.join(x.split(' ',1)[-1] if x.startswith(('LOAD_FAST','LOAD_GLOBAL','LOAD_ATTR','LOAD_CONST','STORE_FAST')) else x.split(' ')[0] for x in t)
    print(s[:3200])
sl=load_original('core.slots').numpy(); print('core.slots', sl.shape, 'norm per slot', np.round(np.linalg.norm(sl.reshape(-1,192),axis=1),2))
```