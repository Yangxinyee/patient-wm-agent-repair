`run_stay` takes a bundle list, not a stay id, so that call failed harmlessly.

Thinking about the run-281 numbers, I now have a concrete explanation for the v4 failure. The decoder unit's bias was −185 against −39.5 for its gate partner. So the decoder term k·v·x averaged about +146 pre-activation units, while the dose signal riding on it had a spread of only about 0.6. That offset sits inside q, which is LayerNorm'd. A few-percent change in the LayerNorm scale from z draws would then move it by about 5–10 units, enough to flip gates and saturate units. This would happen even for lab and cuff rows, matching what the tool showed.

The fix is to constrain the decoder so its output has near-zero mean within each reading code; the bias then carries the constant. With z-subspace orthogonality on top, only signal × (relative scale change) is left as noise.

Before rebuilding I'll confirm this on the saved v4 decoder:
- the size of its constant part, per reading code;
- the structure of the edit-response (EDR) and decoder (DECR2) dictionaries, which I'll reuse;
- the names of the z_head weights.

A large offset relative to the signal supports the explanation.

```python
for nm in ['DECR2','EDR']:
    D=globals().get(nm); print(nm, type(D).__name__, len(D) if D is not None else None)
    for kk in list(D.keys())[:4]:
        v=D[kk]
        if isinstance(v,dict): print('  ',kk, {a:(tuple(np.shape(b)) if hasattr(b,'__len__') else b) for a,b in list(v.items())[:8]})
        else: print('  ',kk, tuple(np.shape(v)))
print('zhead names', [n for n in NAMES if 'z_head' in n or 'zhead' in n])
kv=W1e[114]-W1e[10]; kn=W1e[0]-W1e[107]; bet=np.asarray(load_original('query_norm.bias'))
X_=np.asarray(Xf)
for nmv,vv,cc in [('ins',kv,0),('ne',kn,2)]:
    o=X_@vv; print(nmv,'k.v.x per code mean',[round(float(o[cfa==c].mean()),1) for c in range(4)],'sd on-code',round(float(o[cfa==cc].std()),2),'k.v.beta',round(float(vv@bet),1),'|kv|',round(float(np.linalg.norm(vv)),1))
```