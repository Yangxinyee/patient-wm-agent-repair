Three findings from that run:

- The recurrent part has no near-identity memory channel. The per-step retention a_bar is only 0.29–0.43, and c_out is a dense mixing matrix (its diagonal averages −0.015), so h is rebuilt every bundle.
- The four slots of h are effectively identical (relative spread 1e-4), so the state is really 192-dimensional.
- The dose_edit output gives, per stay-hour and edit, the shift, its standard error, the factual predicted mean and a "reached" flag.

So I can't plant a clean dose memory in the dynamics. The most controllable edit is on the head side: z_prior and z_head are used only by the heads, and the value head MLP can be edited as well. Such an edit would read the dose memory the state already carries and turn it into a shift of glucose or MAP predictions. The rollout's feedback through drawn readings would then carry that shift forward.

To build and test this I need a clean, re-loadable driver that runs `core_step` over a stay with modified doses. I can't read `core_step`'s source, so first I disassemble `step_err` to see exactly how it builds hf, M, ages and the call. If the disassembly shows those conventions, I can reproduce them in a file-based driver. If not, I will infer them by trial against `step_err`.

```python
import dis, io, contextlib
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    dis.dis(step_err)
txt=buf.getvalue()
# compact: keep only opname + argrepr
lines=[]
for ln in txt.splitlines():
    p=ln.split()
    if len(p)>=2:
        # find opname token (all caps)
        toks=[t for t in p if t.isupper() and len(t)>2]
        if toks:
            i=p.index(toks[0]); lines.append(' '.join(p[i:]))
print(len(lines))
print('\n'.join(lines[:260]))
print("types:", type(first_of), getattr(first_of,'shape',None), type(tmin), getattr(tmin,'shape',None), type(dts), getattr(dts,'shape',None))
```