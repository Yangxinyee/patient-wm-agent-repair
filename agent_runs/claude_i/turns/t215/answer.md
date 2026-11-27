The error comes from my own code: `codes_` holds integer code ids, not code names, so the name-to-id lookup failed. I will convert the ids back to names and rerun the same check.

The question is unchanged. I want to know whether `attr_heads.value_bin.0` has units that never fire on any factual reading, so I can reuse them as gated dose units, and what interface my encoder has for re-encoding drug atoms at edited rates.

```python
ID2C = {v:k for k,v in CID.items()}
cods = [ID2C[int(c)] if not isinstance(c,str) else c for c in codes_]
print('codes present:', {c: cods.count(c) for c in set(cods)})
preH, qH, wcH = head_hidden(ctxF.double(), cods, lvv, np.asarray(lastage, dtype=float))
mx_ = preH.max(0).values.numpy(); deadH = np.where(mx_ < -3)[0]
print('value_bin.0 units max pre-act < -3:', len(deadH), '| < -2:', int((mx_<-2).sum()), '| examples', deadH[:12].tolist())
print('enc2 defs:', [l.strip()[:90] for l in open('/work/enc2.py').read().split('\n') if l.strip().startswith('def ')])
```