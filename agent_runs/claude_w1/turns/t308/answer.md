That explains the failure. The old insulin decoder had a z loading of 56 and a register loading of 58, against a total norm of 112. The norepinephrine decoder and the gate were contaminated too. Rebuilt orthogonal to those 41 directions, the gate is still clean (±0.98) and has no z sensitivity. The decoders keep about the same noise (insulin 1.33, norepinephrine 0.21). I will rebuild edit v2 with the clean gate and decoders at the same gains. A tool factual check then shows whether the blowup is gone, and how the tool's real cost compares with the replica's estimate.

```python
gw_old=gw.copy(); gw=gw2.copy()
Le=build(-2.8,3.0,DEC3[('ins',3.6e-5)],DEC3[('ne',3.6e-5)],save=True)
```