Averaging over 64 prior draws barely helps (RMSE 2.03 → 1.96), and every extra flag column makes the fit worse. So the replica is right in structure but leaves about ±2 mg/dL of level error, which I can live with for gradients. It shows that the finger value head depends on the state only through the shared `view` vector, and on the register through columns 0, 3 and 6.

The real crux is the other half of the chain: how an injection into the prelude residual changes the new slots y. My stored Jm has unclear semantics, and its source code is lost. Rather than guess, I'll recover the logic of my CPU step replica (`step_full`, which takes a `dE` perturbation) and related helpers from their compiled code objects: argument names, globals used, constants. That will tell me what Jm's input space is and how to call the replica correctly.

If `step_full` adds `dE` before or after `prelude_norm` in a recognisable way, I can recompute a trustworthy Jacobian from it. If not, I'll measure the response empirically with clean probe edits through the states tool.

```python
import dis
for nm in ['step_full','step_parts','coda_only','out_from_y','build_v3','prelude','prelude_tail','vproj','mindelta']:
    f=globals().get(nm)
    if f is None: continue
    co=f.__code__
    print('==',nm,'args',co.co_varnames[:co.co_argcount],'defaults',f.__defaults__)
    print('   locals',co.co_varnames[co.co_argcount:co.co_argcount+30])
    print('   globals',co.co_names[:60])
    print('   consts',[c for c in co.co_consts if isinstance(c,(str,int,float))][:30])
```