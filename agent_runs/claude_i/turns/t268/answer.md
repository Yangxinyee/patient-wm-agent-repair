No tensor named `core.z_in*` exists, so z_in must sit under another name. The prelude layer has bias-free q/kv/out projections, a 768-unit FFN, and norm1 and norm3.

The z-token probe means dose information has to reach the slots through attention to the u token. The cleanest way to then combine the actual rate (from u) with the expected rate (from the slot's glucose belief) is inside the prelude FFN. It reads each slot's residual after attention and writes straight back into it, with no attention bottleneck on output. If a few FFN units are dead on slot tokens, they can compute a gated deviation and write a "glucose lower" direction.

Next I'll check the head count and output format of `attnw`, find the z_in tensor, and measure the attention weights from the slots to the e, u and z tokens at insulin bundles. A sizeable weight on u means the rate can be read after attention. A tiny one means even u barely reaches the slots.

```python
print([n for n in NAMES if 'z_in' in n or n.startswith('z_')][:10])
xq = torch.randn(4,192); xkv = torch.randn(7,192)
try:
    r = attnw(xq, xkv, 'core.prelude.0.attn'); print('attnw out', [tuple(t.shape) for t in r] if isinstance(r,(tuple,list)) else tuple(r.shape))
except Exception as ex: print('attnw ERR', repr(ex)[:200])
```