Run 218 gave a pooled factual cost for the current edit (N5: dose and rate columns of the numeric embedding x5) of +0.0021 nats per reading, standard error 0.0008. That is well inside the 0.005 budget. By event the cost was: glucose_lab +0.006, map_arterial +0.004, glucose_finger +0.002, map_cuff about 0. My notes stop at pass 30, so I don't know what pass 31 was or what the trailing 4-vector is.

Next I will recover that context: the current operations, the contents of fwd_31, and the code of run 218 if the kernel kept its input history. If the context matches my notes, ops 1-2 are N5 and pass 31 was another factual check of N5. Then N5 stays as the safe fallback while I decide whether any remaining lever is worth the passes. If it doesn't match, I first correct the record.

```python
import os
print(ops())
d='/work/fwd_31'; print([(f, os.path.getsize(os.path.join(d,f))) for f in sorted(os.listdir(d))])
try:
    src=In[-2]; print(len(In)); print(src[-1200:])
except Exception as e:
    try:
        print(_ih[-2][-1200:])
    except Exception as e2:
        print('nohist', type(e).__name__)
```