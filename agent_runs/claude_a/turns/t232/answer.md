`core.delta_in` is a constant per-dimension vector, not computed from the input. So the state-update gate (`abar`) and write scale are fixed per dimension, and a memory dimension cannot be reset by bundle type. A dedicated memory dimension can still persist with a chosen time decay (`flow.a`, `a_in`). The write must then be gated elsewhere (prelude FFN units), and the readout must compensate in the head.

Since only the four mean readouts are scored, a simple time-decaying "recent insulin" memory with a calibrated readout is enough. I will tune the decay and gain against my estimated targets. To build it I need my existing `kill()` helper, which zeroes a dimension's reads and writes. Next I'll print lib1.py (30 lines) to see what it handles.

```python
L=open('/work/lib1.py').read().splitlines()
for i,l in enumerate(L): print(i+1, l[:150])
```