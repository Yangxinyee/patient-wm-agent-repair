The atom table `AT` (stay, hour, event, kind, value, rate, dose, minute, bundle row) is the input to `encode_bundles`, and `atom_vectors` builds each atom's embedding from the code, kind, unit, bins and numeric terms.

To roll the model forward offline, I need to re-encode bundles with sampled readings and edited doses, and to evaluate the value head myself in batch. This run will:
- call `encode_bundles(AT, rows)` and check it reproduces `eALL`/`uALL` and how fast it is;
- list the head-side tensors (`last_*`, `attr_heads.*`, `embed.*`) and the code index and bin-centre tables, so I can write a batched head.

An exact match would confirm I can reuse the encoder as a black box for building fixed atom vectors. A mismatch means its arguments differ from what I assume.

```python
t0 = time.time()
out_enc = encode_bundles(AT, list(range(len(SB))))
print("encode_bundles returned", type(out_enc), len(out_enc) if hasattr(out_enc,'__len__') else '', "time", round(time.time()-t0,2))
if isinstance(out_enc, tuple):
    e_t, u_t = out_enc[0], out_enc[1]
    print("e err", float((e_t-eALL).abs().max()), "u err", float((u_t-uALL).abs().max()))
for pre in ['last_','attr_heads','embed.','filter','pending','time_head','count_head','code_bias','monitor','cont_head','delay_head','exec','status']:
    ks = [k for k in NAMES if k.startswith(pre)]
    if ks: print(pre, '|', '; '.join(f"{k}{list(SHAPES[k])}" for k in ks))
print("cid", cid); print("codes_names", codes_names)
print("VOCAB keys", list(VOCAB.keys()))
print({k: (VOCAB[k] if not isinstance(VOCAB[k], (dict,list)) or len(str(VOCAB[k]))<400 else str(VOCAB[k])[:400]) for k in VOCAB if k in ('norm','normalisation','normalization','codes')})
```