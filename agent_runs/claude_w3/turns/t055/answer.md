tensors.json only lists shapes, so I'll have to infer the categorical indexing from the states, which is expensive.

There is a cheaper question to settle first: how much factual damage would a simpler edit do? The original model already predicts the average post-insulin fall from the glucose level, so an edit that adds the full dose effect would double-count. I can price that from the fwd_2 factual predictions, approximating each prediction as a Gaussian whose spread I estimate from the residuals. I'll compare three shifts of the glucose prediction:
- (a) the full level effect Σθ_L·dose;
- (b) its increment since the last reading of that code;
- (c) the increment computed on jitter deviations only.

If (b) costs well under 0.005 nats per reading overall, a simpler design is acceptable. If only (c) is cheap, the edit must compare the dose with the band.

```python
nm_list = VOCAB['numeric']; print([ (e.get('code', e.get('name')), list(e.keys())[:8]) for e in nm_list][:3])
gfe = [e for e in nm_list if str(e.get('code', e.get('name'))) in ('6','glucose_finger')]
print(json.dumps(gfe[0])[:400] if gfe else 'no gf entry')
```