The notebook gives the targets I'm aiming for, estimated from the records: insulin x2 about -13.1, insulin x1/2 about +6.5, norepinephrine x2 about +4.7, norepinephrine x1/2 about -2.4. The original model gives -1.85, +0.78, -0.50 and -0.60. The current combo is over the factual budget, so it has to be weakened or replaced. Next I'll re-read the architecture in a short chunk, to look again at how the action token reaches the state and where a selective gain could sit.

```python
a=open('/model/ARCHITECTURE.md').read()
print(len(a))
print(a[:3000])
```