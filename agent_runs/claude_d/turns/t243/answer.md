The prediction was wrong. MAP readings that directly follow a norepinephrine bundle do not carry more decodable rate information. For arterial MAP readings with norepinephrine in the previous hour (2,162 rows):

| Target | R2 from full state | R2 from head input |
|---|---|---|
| Rate (mean 0.52, sd 0.31) | 0.56 | 0.42 |
| Dose surprise D (sd 0.25) | 0.32 | 0.18 |

The decoder slope from the head input is k ≈ 0.19. The model's residual slope on D is +2.05, so its factual fit also misses part of the norepinephrine effect.

A rough calculation shows the trade-off:
- A head channel with the full counterfactual gain (about 3 mmHg per unit, i.e. 16x the decoder's slope) would cost about 0.005 nats overall. That is too much on its own.
- About two-thirds of that gain would cost about 0.003 nats.
- Half gain would cost about 0.001 nats.

So a partial norepinephrine correction looks affordable, and the dose_edit tool can calibrate it.

Next I will set up the pieces for a bf16-safe gated norepinephrine channel on hidden units 54 and 9, mirroring E3-ins:
- check the return format of my offline evaluator `eval_E`;
- confirm that WCd holds the 4 z-draw head inputs;
- get the event names and the MAP bin centres for the tilt.

```python
W1c,b1c,W2c,b2c = [np.asarray(load('attr_heads.value_bin.%s'%s)) for s in ['0.weight','0.bias','2.weight','2.bias']]
r = eval_E(W1c, b1c, W2c, b2c)
print(type(r).__name__, len(r) if hasattr(r,'__len__') else r)
if isinstance(r,(tuple,list)): print([(type(x).__name__, getattr(x,'shape',None)) for x in r])
if isinstance(r,dict): print(list(r)[:20])
print('WCd~WC0 corr', round(np.corrcoef(WCd.mean(0)[:,0], WC0[:,0])[0,1],3), round(float(np.abs(WCd.mean(0)-WC0).mean()),3))
print(RD2.event.value_counts().to_dict())
cods = VOCAB['codes']; print(type(cods).__name__, cods if len(str(cods))<300 else str(cods)[:300])
vc = VOCAB['value_bin_centres']; print(type(vc).__name__, list(vc)[:8] if isinstance(vc,dict) else np.shape(vc))
```