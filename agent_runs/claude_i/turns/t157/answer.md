Every `step_err` call raised an exception. My loop swallowed them, so I don't yet know why: a missing global, a wrong argument type, or a changed default. I will call it once on a single bundle and print the full traceback. That shows which input it expects or which global is missing. If it is a simple argument or global issue I can fix it and get the error statistics. If the helper is broken in a deeper way, I will rewrite the step function myself, modelled on the names and constants in its code object.

```python
import traceback
try:
    r0=step_err(5)
    print("ok", r0)
except Exception as ex:
    traceback.print_exc()
try:
    r1=check({'incl': 1, 'unit': 's', 'age': 'both'}, kmax=60)
    print("check ok", type(r1), r1 if not hasattr(r1,'shape') else r1.shape)
except Exception as ex:
    traceback.print_exc()
print(type(dts), getattr(dts,'shape',None), type(first_of), type(tmin), type(Ee), getattr(Ee,'shape',None), type(Uu), getattr(Uu,'shape',None), type(ZZt), getattr(ZZt,'shape',None), type(WT))
```