The tensor list confirms the structure I noted, with a few useful details:
- `core.delta_in` and `core.a_in` are plain per-dim vectors. So a memory dim can be made exactly non-decaying, apart from the chosen `flow.a` time decay, without any input-dependent gate.
- Attention layers have no biases.
- The value head is a 192→192 GELU→17 MLP plus the per-(code, last bin) table `embed_transition.value_bin` (221 = 13 codes × 17 bins). The table can absorb any constant offsets my readout units add.
- `embed.code` and `embed.bins.*` are shared with the encoder, so I must not edit them. `last_numeric`, `last_status`, `type_emb`, `query_norm`, `gap` and `z_head` are head-side and editable.
- W and Wt are identical to the original P. `psi_ins` sums to about −13.2 over 10 lags and `psi_ne` to about +9.2 over 8 lags.

My plan is three dedicated h dims for insulin:
- a memory dim, which never decays through `a_in` and is cleared of all other readers and writers;
- a staging dim, where prelude head 3 writes the decoded insulin rate from the u token;
- a zero-content reference dim, which cancels LayerNorm common-mode shifts at the write and at the readout.

The readout is a pair of repurposed value-head units: one reads the memory, the other is an identical copy without it. Their difference gives a code-gated linear logit tilt for glucose only.

Before choosing dims and units, I need three things:
- my earlier per-dim cost file (`/work/dimcost.npy`, presumably the cost of repurposing each dim);
- the code list, to know which codes are glucose and which are MAP readings;
- the internals of `step_b`/`tf_states`, so I can extract attention weights and intermediate activations for fitting the decoders.

`getsource` fails, so I'll list the code objects' variable names and constants instead.

```python
dc = np.load('/work/dimcost.npy'); print('dimcost', dc.shape, np.round(np.percentile(dc,[0,5,25,50,75,95,100]),5))
o_ = np.argsort(dc); print('cheapest dims', o_[:12].tolist(), np.round(dc[o_[:12]],5))
print('codes', VOCAB.get('codes') if isinstance(VOCAB.get('codes'),list) else list(VOCAB['codes'])[:20] if 'codes' in VOCAB else list(VOCAB.keys()))
print('f2 cols', list(f2.columns)); print(f2.iloc[:,:].head(3).to_string()[:600])
for fn_ in [step_b, tf_states, MH, rollout_gen, head_value]:
    co = fn_.__code__
    print(fn_.__name__, 'args', co.co_varnames[:co.co_argcount], '| locals', co.co_varnames[co.co_argcount:][:60], '| names', co.co_names[:60])
```