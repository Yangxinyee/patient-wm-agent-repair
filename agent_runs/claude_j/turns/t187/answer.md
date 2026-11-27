The head details are now pinned down. Glucose_finger and glucose_lab have different 16-bin centres, `last_numeric` is Linear(8→192), and the predicted mean averages the centres over bins 1..16.

My edit will use a dead hidden unit of `attr_heads.value_bin.0`, held in GELU's linear regime and switched on only for glucose codes (another unit for MAP codes). Its output tilts the bin logits, and its constant part is cancelled in that code's `embed_transition` rows.

The hard part is gating by code without adding state-dependent noise. Because `query_norm` output satisfies Σ(q−β)/γ = 0, the direction 1/γ gives a constant w·q. So gating can rely only on each code's projection p_c = Σ E_code[c]/γ.

I will compute p_c for all 13 codes, plus the spread that the last-value terms add along that direction. If glucose codes and MAP codes separate cleanly, this gate works.

```python
gq=HT('query_norm.weight'); bq=HT('query_norm.bias'); Ec=HT('embed.code.weight'); Wln=HT('last_numeric.weight'); bln=HT('last_numeric.bias'); Bv=HT('embed.bins.value.weight')
ig=1/gq
print('min|gamma|',float(gq.abs().min()))
names_=[c['name'] for c in sorted(VOCAB['codes'],key=lambda c:c['id'])]
print({n[:10]:round(float(Ec[i]@ig),2) for i,n in enumerate(names_)})
print('Wln cols proj',[round(float(Wln[:,j]@ig),2) for j in range(8)],'bln',round(float(bln@ig),2))
print('bins.value proj range',round(float((Bv@ig).min()),2),round(float((Bv@ig).max()),2), 'const q', round(float(bq@ig),2))
```