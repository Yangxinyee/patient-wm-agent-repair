My notebook confirms the key data fact: within a glucose tier, the insulin rate varies by independent random noise. So the record-based effect (about -2 mg/dL per U/h at lag 1, cumulative about -11) is unconfounded. The original model's factual slope on that noise is about 0, so its factual predictions ignore real dose information.

The notebook doesn't describe pending_mlp, filter_gate, last_numeric, gap, query_norm or embed_transition. If any of them feeds dose information to the prediction heads without passing through the action token's two LayerNorms, it would be a lever with no saturation.

Next I'll pull only the ARCHITECTURE.md lines that mention these components, truncated to keep the output short. If such a route exists, it becomes my next test. If all of them take the normalised action token as input, I conclude that N5 is close to the ceiling.

```python
A=open('/model/ARCHITECTURE.md').read().split('\n'); print(len(A))
keys=['pending','filter','last_numeric','gap','query','transition','coda','flow','delta','a_in','age','slots','prelude']
out=[]
for i,l in enumerate(A):
    if any(k in l for k in keys): out.append(f"{i}: {l[:170]}")
print('\n'.join(out)[:3300])
```