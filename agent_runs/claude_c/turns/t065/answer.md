Picking up after the restart. Run 57 showed that the dose-induced part of the model's readout vector is small next to the natural variation (about 0.03 to 0.08 of it per coordinate) and is spread over many coordinates. The kernel now holds 30 rank1 ops on `embed.bins.dose.weight` and `embed.bins.rate.weight`. The notebook only describes the earlier `embed.numeric` ops, so the last session's work was not written down. Before I change anything I need to recover that context: the architecture, the full notebook, the current ops and the kernel variables. If the ops turn out to be an unfinished experiment, I will judge them against a new forward pass rather than assume they help.

```python
import os, json
print(open('/model/ARCHITECTURE.md').read()[:9000])
print("=====NOTES=====")
txt=open('/work/notes.md').read()
print(len(txt))
print(txt[:6000])
print("=====OPS=====")
try:
    print(ops())
except Exception as e:
    print('ops err',e)
print("=====GLOBALS=====")
g=[k for k in globals().keys() if not k.startswith('_')]
print(g)
print(os.listdir('/work/vectors'))
```