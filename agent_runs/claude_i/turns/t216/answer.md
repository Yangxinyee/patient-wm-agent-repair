Only one unit of `attr_heads.value_bin.0` is fully inactive on factual readings, and three are nearly so. That is enough for one gated unit per drug, provided I cancel the unit's constant offset. I can do the cancellation exactly through the per-code transition table `embed_transition.value_bin`, which is added to the logits for each code separately. This also gives clean gating: the insulin unit acts only on the glucose codes and the norepinephrine unit only on the MAP codes.

The size of the readout should come from the data. In factual records, the model's prediction errors should be explained by past departures of each dose from what the policy would have given, with some kernel J. The tool's teacher-forced predictions for 100 training stays (`F`) let me estimate J without spending a forward pass.

I need three things for that:
- the column layout of `F`;
- my policy-expectation function (`pibar`);
- the drug-atom format my encoder expects, so I can later re-encode edited rates.

The estimate is promising if the regression of glucose_finger prediction errors on lagged insulin deviations comes out clearly negative. If it is near zero, the model's teacher-forced predictions already capture the dose effect, and the problem lies only in how it extrapolates to edited doses.

```python
print('F cols', list(F.columns)); print(F.head(2).to_string()[:500])
try: print('pibar', [round(float(v),3) for v in np.ravel(pibar(np.array([140.,160,175,190,210,230,260,300])))])
except Exception as ex: print('pibar err', repr(ex)[:200])
src = open('/work/enc2.py').read(); i0_ = src.find('def emb_atoms'); print(src[i0_:i0_+1300])
```