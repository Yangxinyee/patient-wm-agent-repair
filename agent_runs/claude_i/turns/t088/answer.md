out equals core.out_norm applied to the mean of the h slots, to 1e-6. So the state layout and LayerNorm convention are confirmed. The h slots are large (norm about 509), out is about 14, and the z posterior mean has sd 0.26. The earlier z-posterior error of 0.047 is therefore about 18% of its spread, which is not good enough. Before writing a clean step implementation, I'll look at the rest of enc.py to recall the encoder conventions I settled earlier: END atoms carrying attributes, the kq index, and the size term.

```python
s_enc = open('/work/enc.py').read()
print(s_enc[1500:])
```