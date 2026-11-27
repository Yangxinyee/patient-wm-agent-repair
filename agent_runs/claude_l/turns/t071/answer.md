The source of my earlier functions is lost, because they were defined in the kernel without a saved file. The tensor list does clarify the core.

- **Coda.** It is a pre-norm transformer layer with self-attention and an FFN, plus residual connections and no final norm. So `h` is a residual stream and is not renormalised each step. That is good for a persistent memory channel.
- **c_out.** It is a full 192×192 linear map with no bias.
- **Recurrent block.** It is two layers with self-attention, cross-attention to memory and an FFN.

**Conclusion:** a dedicated slowly decaying dose-memory channel in a few dimensions of `h` is architecturally possible. To build it, I need to zero those dimensions' rows in the residual write-backs, make `c_out` act as identity there, set the input via `b_in` and the decay via `flow.a`/`a_in`, then add a readout in the value head.

To design and test this without spending forward passes, I need a full, exact sandbox copy of the step and the heads. This time I will write it into a file (/work/wm.py) so it survives restarts.

Next I will recover the details I worked out before (config such as the number of heads, and the signatures and referenced names of my old functions) and list what forward passes 1–2 returned. That tells me what I can verify against.

```python
import dis, os, json
print(json.dumps(VOCAB.get('config',{}))[:1500])
print([k for k in VOCAB.keys()])
for fn in ['atom_vec','set_encode','build_bundles','atoms_tensor','z_pred_k','LN','mlp','lin','ln','atomvec','setenc','mlp2l','zscore','fbin','fb_le','empty_enc','variant_atoms','run_combo','setenc_v','zerr','feats','LNt','T64']:
    f=globals()[fn]
    try:
        sig=str(inspect.signature(f))
    except Exception as ex: sig='?'
    c=f.__code__
    print(fn,sig,'| names:',c.co_names[:40],'| consts:',[x for x in c.co_consts if isinstance(x,(int,float,str)) and not (isinstance(x,str) and len(x)>40)][:25])
for k in ['fwd_1','fwd_2']:
    print(k,os.listdir('/work/'+k))
```