The cached datasets store e and u precomputed (13k bundles × 192), so an atom-encoder cut would not show up in `tf_eval` until I re-encode. The encoder helpers (`atom_vectors`, `encode_bundles`, `deepsets`) take no params dict, so they use a global.

Before paying for encoder cuts and re-encoding, I want to test a cheaper code gate. It would read the value MLP input, `with_code`, along a linear direction that separates glucose_finger queries from other codes. That works only if the code embedding terms dominate the state-dependent variation from q. Testing it needs explicit access to `with_code` per reading.

So next I will:
- print the head paragraph of ARCHITECTURE.md, to get the exact inputs of `last_numeric` (8 columns) and of `view`, `z` and `gap`;
- print the signatures of `tf_views`, `head_probs`, `head_mean`, `tf_states` and `ctx_rows`.

If `tf_views` already returns per-reading views, I can build q and `with_code` directly. Otherwise I will write an explicit head replica and check it against `head_value`.

```python
i=A.find('**Heads'); j=A.find('with_code')
print(A[A.find('view', i-900 if i>900 else 0)-200: A.find('view', i-900 if i>900 else 0)+700].replace('\n',' '))
print('====')
print(A[j-50:j+900].replace('\n',' '))
for f in ['tf_views','head_probs','head_mean','tf_states','ctx_rows','flag','true_shift']:
    try: print(f, str(inspect.signature(globals()[f]))[:120])
    except Exception as ex: print(f,'ERR',ex)
```