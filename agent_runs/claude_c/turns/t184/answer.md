The core step (step 3) shows the route: slot tokens h+slots, the reading token e_k, the action token u_k and the z token all go through one pre-norm attention layer. Only the slot outputs are kept, and they are LayerNormed into E. Because h has norm about 842, anything attention adds is divided by about std(h) inside `prelude_norm`. That explains the weak, non-selective levers. I now need the Input section (lines 7–30), to see how u_k is built and what it is at bundles without a drug. A constant "no-drug" u_k would explain why amplifying head 0 costs factual fit.

```python
print('\n'.join(x[:160] for x in L[7:30]))
```