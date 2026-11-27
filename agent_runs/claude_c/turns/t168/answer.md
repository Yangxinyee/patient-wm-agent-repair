Scaling head 0 ×4 raised the insulin readouts to −2.53 / +1.67, from −1.85 / +0.78. The half-dose response roughly doubled, already better than scaling all heads ×3 (−2.33 / +0.89). NE moved to −0.36 / −0.97, still the wrong sign for ×2. So head 0 carries part of the dose pathway but well under ×4 proportionality. I'll test head 1 the same way to find which head carries most of it.

```python
clear_ops(); scale_head(1, 4.0); print(len(ops()))
```