Two things from the last output change the wiring plan:
- `kill()` already cuts a dim out of every core writer and reader, and out of the head's `type_emb` and `query_norm`.
- The value head is a 192-unit MLP giving 17 bin logits, plus a per-(code, last bin) logit bias from `embed_transition`.

The hard part is gating the readout to glucose_finger only. `embed.code` is probably shared with the atom encoder, so I cannot put a code-gate constant there. The candidate is `type_emb`, which is head-only and added before `query_norm`; it works only if the value query's "type" index is specific to the code.

Next I will grep ARCHITECTURE.md for how `type_emb`, the value bins and fine bins, and `embed.kq` are used. I will also look at VOCAB's structure, including the bin centres for glucose_finger.

How I will read the result:
- If `type` is indexed per code (13 codes + 2), `type_emb` can carry the gate.
- If not, I need another head-only per-code channel.

```python
for kw in ['type_emb','value_bin','fine','kq','with_code','type ']:
    i=A.find(kw); print(kw,'|',A[max(0,i-120):i+230].replace('\n',' ') if i>=0 else 'NONE'); print('--')
print(list(VOCAB.keys()))
```