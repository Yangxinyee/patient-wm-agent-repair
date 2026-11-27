The architecture description is short and matches my notes. Doses enter only through the action atoms: z_dose/z_rate in embed.numeric and the dose/rate bins. They pass through encode_action to give u_k, which feeds z_post and the prelude. The main candidates for my re-implementation error are details the description leaves vague. These include the pre-norm layer layout, the attention head count, what the norm is, how "size" and "gap" are computed, and how the 4 slots are combined. Next I will look at my saved /work/impl.py and the kernel variables, to see what I implemented and where it might differ. If impl.py already covers all these steps, the error is in an exact detail such as a norm type or head split, which I can isolate step by step against the states.

```python
src = open('/work/impl.py').read()
print(len(src))
print(src[:9000])
print([k for k in globals().keys() if not k.startswith('_')][:200])
```