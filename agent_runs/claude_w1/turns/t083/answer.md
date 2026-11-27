The listing settles several replica details:

- **Prelude and coda:** both are pre-norm layers with self-attention and an FFN only (norm1 and norm3). Only the two recurrent block layers also have cross-attention to the memory (norm2).
- **Projections:** none of the attention projections have biases.
- **Config:** 4 heads (48 dims each), an FFN width of 768, and a memory of 128 entries.
- **Core step:** the remaining core-step parameters are the time-flow rates, the input gate (delta_in, a_in, b_in), c_out and the slot embeddings. Nothing hidden is missing.

With the latent posterior already replicated exactly, a one-step core replica is now worth retrying. I will test it teacher-forced: take the tool's saved state after bundle k-1, compute bundle k, and compare with the saved state after bundle k.

First I need to recall what my existing helpers do (encoder, context, layer norm), so I can reuse them instead of rewriting them. I also need the layout of the saved state file.

What to look for in the output:
- **Good sign:** the helpers' sources and the state file show everything needed (per-bundle times, bundle types, previous states), so the replica can be built in the next run.
- **Missing pieces:** for example, if bundle times are not stored, I will recover them from the records first.

```python
import inspect
for fn in ['LN','mlp','lin','enc_set','enc_set2','encode_set','atom_e_full','ctx_','cur','head_P2','btype','block_stats']:
    obj = globals().get(fn)
    if obj is None: 
        print('##', fn, 'missing'); continue
    try:
        src = inspect.getsource(obj)
        print('##', fn, '\n', src[:900])
    except Exception as ex:
        print('##', fn, type(obj), str(ex)[:80])
S3 = np.load('/work/fwd_3/states_original.npz', allow_pickle=True)
print({k:(S3[k].shape, S3[k].dtype) for k in S3.files})
```