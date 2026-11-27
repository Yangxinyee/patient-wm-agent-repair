Scaling head 1 ×4 weakened the insulin response (−1.50 / +0.45, vs the original −1.85 / +0.78) and left NE about where it was (−0.37 / −0.52). Head 1 doesn't carry the action signal. Amplifying it seems to dilute the other heads' contributions after the prelude normalisation. Head 0 remains the only head so far that strengthens the dose response. I'll test head 2 the same way: readouts above the original would show it carries dose signal, readouts below would show it is another diluting head.

```python
clear_ops(); scale_head(2, 4.0); print(len(ops()))
```