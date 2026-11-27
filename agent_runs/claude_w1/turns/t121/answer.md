The tensors are confirmed. The value head is `attr_heads.value_bin` (192→192→17), and `embed_transition.value_bin` is a 221×17 table (13 codes × 17 last-bin states). The tuning records are in `tu`, and `fwd_2` has the tool's per-reading predicted means for the 100 setA stays.

Next I'll compute predicted means myself from `gen_samples`' head-input vectors and check them against the tool. The computation is: softmax over the head MLP output plus the transition row, then Σ p·centre. An MAE well under ~1 (mg/dL or mmHg) means I can compute factual residuals locally for thousands of training stays without using forward passes. First I need the vocab layout for codes, edges and centres.

```python
print(list(VOCAB.keys()))
for k in VOCAB:
    v = VOCAB[k]
    if isinstance(v, dict):
        print(k, '-> dict keys', list(v.keys())[:6], '...' if len(v) > 6 else '')
    elif isinstance(v, list):
        print(k, '-> list len', len(v), str(v[:3])[:120])
    else:
        print(k, '->', str(v)[:80])
```