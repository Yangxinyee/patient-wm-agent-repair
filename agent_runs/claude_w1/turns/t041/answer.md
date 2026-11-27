I'm restarting from the notebook. It covers fwd 1, but three forward passes have been used, and fwd 2 and fwd 3 aren't written up. My last run (37) was an attempt to reproduce the model's forward pass locally. The reference "z after bundle0" didn't match any variant I tried: the best error was 1.79, so my reimplementation is still wrong somewhere.

Before going further I need to recover context:
- the architecture, which I'll re-read;
- what fwd_2 and fwd_3 contain;
- which variables the kernel holds.

If my reimplementation can be made to match the states the tool returns, I can test weight edits locally. If it can't, I'll design the edit analytically and check it with forward passes.

```python
import os, json, glob
print(open('/model/ARCHITECTURE.md').read())
print("=====")
for d in sorted(glob.glob('/work/fwd_*')):
    print(d, os.listdir(d))
    for f in os.listdir(d):
        p=os.path.join(d,f)
        print('   ', f, os.path.getsize(p))
print("=====")
g={k:type(v).__name__ for k,v in globals().items() if not k.startswith('_') and type(v).__name__ not in ('module','function','builtin_function_or_method','type')}
print(g)
print([k for k,v in globals().items() if callable(v) and not k.startswith('_') and type(v).__name__=='function'])
```