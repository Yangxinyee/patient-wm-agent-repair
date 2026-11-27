The kernel holds a concrete role assignment for h dims: `ROLE = {jR:132, jG:52, jWI:73, jSI:105, jWN:144, jSN:128, jI:155, jN:171}`, with memory dims `MEMD = [155, 171]`.
- **Plausible meanings:**
  - jR: reference dim.
  - jG: glucose-compensation staging.
  - jWI and jWN: write-staging dims for insulin and NE.
  - jSI and jSN: second-stage dims.
  - jI and jN: memory dims for insulin and NE.
- **Rollout helper:** `rollout_gen` takes dose specs and also an optional per-code value "shift". I probably used that to calibrate how a per-step tilt turns into a rollout effect.
- **Bug in my last run:** it failed because I had loaded the dim-cost file as `dc`, not `dimcost`.

I can't read source code, so next I'll inspect the code objects of the scaffold-building functions: variable names, referenced globals and constants. This tells me what `scaffold_v`/`make_scaffold` edit and what format `flags` takes, so I can reuse them instead of rebuilding. If the constants show tensor names such as `flow.a`, `a_in`, `b_in`, `c_out` and value-head names, the scaffold implements design v2.

```python
def cinfo(f, n=60):
    c=f.__code__
    consts=[x for x in c.co_consts if isinstance(x,(str,int,float)) and not (isinstance(x,str) and len(x)>60)]
    print(f.__name__,'| vars:',c.co_varnames[:c.co_argcount+25])
    print('  names:',c.co_names[:n])
    print('  consts:',consts[:n])
for fnm in ['make_scaffold','scaffold_v','f_','evalf','score']:
    cinfo(globals()[fnm])
print('rollout_gen doc:', rollout_gen.__doc__[:400])
for r in range(3):
    d=dc[r]; o=np.argsort(d)
    print('dc row',r,'mean %.4f'%d.mean(),'best',o[:8].tolist(),np.round(d[o[:8]],4).tolist(),'roles',[round(float(d[j]),4) for j in JALL])
```