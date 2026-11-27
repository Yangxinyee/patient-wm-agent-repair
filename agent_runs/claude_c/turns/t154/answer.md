Zeroing the action encoder's output offset left all four readouts essentially unchanged (−1.81, +0.76, −0.53, −0.61 vs the original −1.85, +0.78, −0.50, −0.60). So the offset is not what limits the action signal.

That raises a basic question: does the action token carry the model's dose response at all? I'll also zero the encoder's output gain, which makes the action token identically zero. If the readouts then collapse to about 0, the action token is the carrier and its downstream gain is the bottleneck. If they persist, the signal enters somewhere else.

```python
scale('encode_action.norm.weight', 0.0)
print(ops())
```