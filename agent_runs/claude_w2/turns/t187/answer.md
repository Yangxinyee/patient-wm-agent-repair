The tensor list confirms the routes. Each step multiplies h by about 0.37 (abar 0.31–0.46), so persistence comes from the blocks rewriting it. The time-based flow decay is weak (median 0.98 per hour). There are several other heads (time, count, status, rate, cont, …) whose log loss is only reported. The binding constraint is the value NLL.

So the dedicated-coordinate design is:
- Set coordinate i of h to abar = 1 and c_out[i,i] = 1, with no writes from the blocks or coda, so it decays only through flow.a[i].
- Inject the dose at drug bundles through prelude attention head 2, which puts about 100% of its attention on the u token. A new readout row of prelude attn.out goes into an isolated slot coordinate j, then prelude_norm, then b_in[i,j].
- The new value-head unit is the only reader.

Two facts decide feasibility:
1. Can the dose be read linearly from head 2's 48-dim value of LN1(u)? Is the readout about 0 for empty and NE-only u?
2. How variable are the slot-token LayerNorm statistics μ/σ, which scale the injection?

To compute either I need the exact attention head layout, so next I print my attention and core-step code.

```python
s=open('/work/impl.py').read().split('\n')
print('\n'.join(s[104:159]))
```