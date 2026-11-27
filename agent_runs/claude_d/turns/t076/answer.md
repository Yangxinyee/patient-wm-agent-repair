Restricted to the 1,237 true insulin-dosing bundles, the picture is:

| Target | z (32) R² | out R² | h slots R² |
|---|---|---|---|
| Dose surprise | 0.00 | 0.12 | 0.31 |
| Rate | 0.29 | 0.62 | 0.71 |
| Policy base | 0.48 | 0.88 | 0.88 |

So z carries no dose surprise, and a pure `z_in` edit would do nothing. The h state holds the base level well and the actual dose only weakly.

The one place where context and action meet before the state update is `z_post.0`, which reads [ctx, e_k, u_k]. A centred feature could be built there, roughly the rate decoded from u_k minus the expected rate decoded from ctx. That needs exact numpy versions of the embedding, `encode_action` and ctx, checked against the recorded z posterior means. I'll start by looking at vocab.json's structure and config, which the embedding needs.

```python
voc=json.load(open('/model/vocab.json'))
for k_,v_ in voc.items():
    s_=json.dumps(v_); print(k_, type(v_).__name__, len(v_) if hasattr(v_,'__len__') else '', s_[:300])
```