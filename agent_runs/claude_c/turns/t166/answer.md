Both signs shrink the response: −1.39/+0.28 with alpha = −302, and −0.92/+0.49 with alpha = +300. Making the action token resemble the slot tokens therefore doesn't raise its attention share, and I'm dropping that idea.

A natural selective basis is still unexplored: the 4 attention heads of the prelude. If one head mostly reads the action token, scaling only its output columns of `attn.out` amplifies the dose signal with little effect on readings. That needs 48 per-column rank-1 ops per head. I'll test head 0 at ×4 and log the result so far.

- A head that carries the dose pathway would push the insulin readouts toward 3–4× their original size.
- An unrelated head would leave them near 1×.

```python
clear_ops()
with open('/work/notes.md','a') as f:
    f.write("- Pass 13: encode_action.norm.bias=0 -> readouts unchanged. Pass 14: action token u=0 -> all readouts exactly 0 (u is sole carrier).\n"
            "- Pass 15 factual u=0: NLL 1.63461->1.63820 (+0.0036); |dpred| gf 1.12, map 0.32; per-U/h lag coef ~0.02-0.16 (truth ~2) -> u potency ~1/13.\n"
            "- Pass 16/17: steer encode_action.norm.bias along mean-h dir (alpha +300/-302): ins x2 -0.92/-1.39 -> weaker both signs. Dropped.\n"
            "- Next: per-head scaling of prelude attn.out columns (48 per head) to find an action-specific head.\n")
def scale_head(hd, c, name='core.prelude.0.attn.out.weight'):
    for j in range(48*hd, 48*(hd+1)):
        e = np.zeros(192); e[j] = 1.0; rank1(name, e, c-1.0, side='in')
scale_head(0, 4.0); print(len(ops()))
```