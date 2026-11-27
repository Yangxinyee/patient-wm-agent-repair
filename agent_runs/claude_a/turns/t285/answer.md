The architecture notes rule out `type_emb` as a code switch. It is applied before the code is predicted, so it can't tell glucose_finger apart from MAP. The only code-specific term in the value head's input is `embed.code[code]`, which is shared with the encoders. Instead I'll build the switch as a learned linear direction: a discriminant on the value head's 192-dim input that separates glucose_finger queries from the other codes. Combined with the two-unit trick, the switch's own fluctuation cancels.

Other facts from this run:
- The ablation set core.out_norm's weight and bias to zero at the reserved dims, so the memory currently reaches no consumer.
- Many head modules read the 192-dim query; I'll zero their columns at the reserved dims once I re-enable those dims.

Before writing the builder, this run checks the remaining details:
- the columns of the per-reading table;
- whether `last_numeric` and `last_status` are head-only tensors (safe to zero at the reserved dims);
- the glucose normalisation scale and bin centres;
- whether the latent-prior and posterior input columns at dims 50/99 are already zero;
- the cost estimates I stored for value units.

```python
rdf=ds_s2['rd']; print('rd', type(rdf).__name__, (list(rdf.columns)[:14] if hasattr(rdf,'columns') else len(rdf)))
ln_=[(k,tuple(PA[k].shape)) for k in PA if k.startswith('last_') or 'transition' in k or k.startswith('gap') or k.startswith('z_')]
print('last/gap/z/trans tensors', [(k.replace('weight','w'),s) for k,s in ln_][:16])
zr=[k for k in PA if k.startswith(('z_prior','z_post')) and PA[k].dim()==2]
print('z first-layer cols 50/99 max', [(k, round(float(PA[k][:,[50,99]].abs().max()),4)) for k in zr if PA[k].shape[1]>=192][:4])
nz=VOCAB.get('normalisation', VOCAB.get('normalization')); print('norm keys', type(nz).__name__, (list(nz.items())[:3] if isinstance(nz,dict) else str(nz)[:200]))
print('VOCAB keys', list(VOCAB.keys()))
print('VU' in globals() and VU, 'dimcost val units?', [k for k in globals() if 'vu' in k.lower()][:8])
```