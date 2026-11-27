The core has a one-layer prelude, two blocks (each with self-attention, cross-attention to memory, and an FFN), a one-layer coda, and the linear maps `b_in` and `c_out`, plus the gates `a_in`, `delta_in` and `flow.a`. The dose can only reach the slot state through the prelude attention, from the slot queries to the `u` token or the `z_in(z)` token.

To measure that gate I'll reimplement the prelude and expose its attention weights. For that I need three things:
- the name of the `z_in` tensor;
- the head-side tensor names;
- the layout of the cached bundles in `bundles_B`, so I can feed them through `cstep`.

```python
zn = [n for n in NAMES if not n.startswith('core.') and not n.startswith('embed')]
print("non-core, non-embed tensors:", len(zn))
print("; ".join(f"{n}{tuple(SHAPES[n])}" for n in zn)[:1800])
k0_ = list(bundles_B.keys())[0]
b_ = bundles_B[k0_]
print("bundles_B key type:", type(k0_), "value type:", type(b_), "len:", len(b_))
el = b_[0]
print("element type:", type(el))
if isinstance(el, dict):
    print("element keys:", list(el.keys()))
    for kk_, vv_ in el.items():
        print("  ", kk_, type(vv_).__name__, (str(vv_)[:150]))
elif isinstance(el, (list, tuple)):
    print("element len:", len(el), [type(x_).__name__ for x_ in el])
    print(str(el)[:400])
```