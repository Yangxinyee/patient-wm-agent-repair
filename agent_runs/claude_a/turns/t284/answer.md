Useful details:
- glucose_finger is code 6 and map_arterial is code 7.
- The ablation set query_norm's weight to zero at dims 50/99/79, so the head currently sees no memory.
- Value-MLP columns 50/99 still read constant per-code terms.
- `embed.code`, `embed.bins.*` and `embed.numeric` are shared with the input encoders, so I must not use them for gating.

The readout needs a code-specific switch that lives only in the head. `type_emb` is the candidate, if it is head-only and indexed by the code being predicted. With it I can use a two-unit trick: unit 1 = GELU(gate + w·signal), unit 2 = GELU(gate), output weights +tilt and −tilt. The gate's fluctuation cancels, and the readout is effectively linear for glucose_finger and zero for other codes. To confirm, this run will:
- grep ARCHITECTURE.md for how `type_emb`, `with_code` and `query_norm` are used;
- find the out-norm tensor and its values at the reserved dims;
- list the head tensors that read the 192-dim query, which I would need to shield from the memory.

```python
txt=open('/model/ARCHITECTURE.md').read().splitlines()
hits_=[l.strip()[:170] for l in txt if re.search(r'type_emb|with_code|last_numeric|query_norm|out_norm|type id|types',l)]
print('\n'.join(hits_[:12]))
on=[k for k in PA if 'out_norm' in k]; print('out_norm tensors', on, [[round(float(PA[k][j]),3) for j in (50,99,79)] for k in on], 'orig', [[round(float(PO[k][j]),3) for j in (50,99)] for k in on])
rd=[k for k in PA if PA[k].dim()==2 and PA[k].shape[1]==192 and not k.startswith(('core.','encode','embed'))]
print('192-col readers:', [k.replace('weight','w') for k in rd][:20])
```