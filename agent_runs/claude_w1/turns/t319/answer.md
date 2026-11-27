Edits are dropped; the model is back to the original.

The architecture text answers the key question. The latent z enters the value head only inside q = query_norm(view + type_emb), which is a LayerNorm output. A LayerNorm output has one exactly invariant direction: g = 1/γ, where γ is the query_norm weight. g·q is the same constant for every row, whatever z is drawn. So the gate g·with_code depends only on the code embedding and the register terms (last bin, age, last value), and my replica can compute it exactly.

The new design (v2):
- Per drug, two hidden units share the same gate. One also carries the dose decoder; the output weight of the second is the negative of the first.
- In the target code group both units sit in GELU's linear regime, so their difference is exactly the decoder term, with no cancellation of large constants. In the other group both are off.
- All magnitudes stay moderate (pre-activations about ±10).

This run checks feasibility. The gate g·x must separate glucose codes from MAP codes, with a margin large relative to the spread within each code. I also check that γ has no near-zero entries, and look at z_head's shape and the stored decoders. A clean separation confirms the plan; interleaved codes would refute it.

```python
qn = [n for n in NAMES if 'query_norm' in n]; zh = [n for n in NAMES if 'z_head' in n]; lnn = [n for n in NAMES if 'last_numeric' in n]
print(qn, [SHAPES[n] for n in qn]); print(zh, [SHAPES[n] for n in zh]); print(lnn, [SHAPES[n] for n in lnn])
gam = np.asarray(load_original([n for n in qn if n.endswith('weight')][0]), dtype=np.float64)
print('gamma |min| %.4f |max| %.4f  n(|g|<.05)=%d' % (np.abs(gam).min(), np.abs(gam).max(), (np.abs(gam) < .05).sum()))
g = 1.0 / gam
print('Xf', Xf.shape, 'codes', np.bincount(cf))
gx = Xf.astype(np.float64) @ g
for c in range(4):
    s = gx[cf == c]
    print('code', c, 'mean %.2f sd %.3f p0.1 %.2f p99.9 %.2f min %.2f max %.2f' % (s.mean(), s.std(), np.percentile(s, .1), np.percentile(s, 99.9), s.min(), s.max()))
print('DEC2 keys:', list(DEC2.keys())[:8])
for k in list(DEC2.keys())[:4]:
    v = DEC2[k]
    print(k, type(v).__name__, (list(v.keys())[:10] if isinstance(v, dict) else getattr(v, 'shape', None)))
with open('/work/notes.md', 'a') as f:
    f.write('\n## Session 8 (runs 269-)\n- fwd_5/6 = factual both with EDIT v1 variants: BROKEN (edited NLL 18.3/16.3; dmu ~+95 glucose, -15 MAP). Cause: big-bias gate not invariant to q/z noise. ALL EDITS DROPPED (run 271).\n'
            '- ARCH: q = query_norm(view+type_emb) (LN) -> g=1/gamma_qn gives g.q = const EXACTLY (z-proof gate). with_code = q + code_emb + last_numeric + bins + last_status.\n'
            '- PLAN v2: per drug 2 units, same gate G*(g.x-g0)+B, unit A adds k*v.x, W2 cols +w/-w -> output = w*k*v.x in on-group (linear regime), 0 off-group. Moderate magnitudes. Calibrate c with tool.\n')
```