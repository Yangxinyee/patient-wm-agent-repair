Run 245 showed that a linear read of the insulin rate from head-2 token values tracks the rate within the data range (0.5–5) but saturates above 5, at about 5.8 for a rate of 10. The norepinephrine read is poor and not monotone. My last four replies were stopped, so from now on I keep replies short.

Next, I will look at my saved implementation files and the kernel function list, so I know what is already built before I edit any weights. If impl4.py already contains the channel construction, I can go straight to editing and calibrating. If it does not, I build stage A, the head-2 free value dimensions.

```python
import types
for f in ['impl.py','impl2.py','impl3.py','impl4.py']:
    s=open('/work/'+f).read(); print('=====',f,len(s), s[:300].replace('\n',' | '))
print('----- impl4 full -----'); print(open('/work/impl4.py').read()[:5000])
fn=[k for k,v in globals().items() if isinstance(v,types.FunctionType) and not k.startswith('_')]
print('FUNCS',fn)
with open('/work/notes.md','a') as fh:
    fh.write("\n## Session 12 (restart 12)\n- Run 245: head2-token rate read: ins linear in data range (.5-5), saturates >5 (~5.8 at 10); ne read poor/non-monotone.\n- 4 replies stopped by safeguard: keep replies SHORT.\n")
```