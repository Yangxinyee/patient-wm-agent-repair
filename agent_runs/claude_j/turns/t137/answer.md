The grep confirms the outline but leaves out details I need: what `coda` is, the shapes of `b_in` / `c_out`, and how u is formed. Lines 36–56 hold the whole step description (about 20 lines), so I'll print just that range, each line capped in length. This tells me whether any part of `h` passes through the update in a near-linear way that could carry a dose signal.

```python
for i in range(35,58): print(i, arch_[i][:230])
```