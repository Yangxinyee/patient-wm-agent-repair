Run 86 confirmed that the head re-implementation and many intermediate objects are still in the kernel:
- the 5116 states from forward pass 2 (`X`);
- 10353 reading rows (`RR`) with precomputed head inputs, hidden activations and logits;
- dose-deviation tables (`DEV`, `TF`, ...) and the training regression design tables (`Dtr2`, `Q`, `Ltr`).

The design I'm converging on is a head-only, code-gated correction to the glucose and MAP value predictions. It is proportional to the drug's dose deviation from what the protocol would have given, read from the state. Its size per lag is Δβ = (true one-step coefficient) − (the model's own one-step coefficient), both from the same regression.

The reasoning: within a tier the dose multiplier is random, so in the factual records this correction only adds the drug effect the model is missing. It should cost almost no factual NLL. Under a dose edit, the deviation becomes large and the correction supplies the missing causal effect. The model's own glucose dynamics then carry it forward.

To do this I need the model's one-step dose coefficients. I'll get them by regressing the model's predicted means on the same regressors as the truth regression. I have predicted means for about 200 tuning stays. First I need to recall what the stored tables contain.

**Test:** if the model's lag-1 coefficient on dose deviation is much weaker than the true −2.0, that confirms the plan.

```python
def cols(name):
    v = globals()[name]
    if isinstance(v, pd.DataFrame):
        print(f"{name} {v.shape}: cols={list(v.columns)[:30]}  idx[:3]={list(v.index[:3])}")
    else:
        print(name, type(v).__name__, getattr(v,'shape',None))
for n in ['DEV','DEVs','TF','I','GF','f3','FA','RR','R','Rh','Ltr','Ltu','Q','Dtr2','D','both','both_fl','St','adm','DE1']:
    try: cols(n)
    except Exception as ex: print(n, 'ERR', ex)
print('EDOSE', EDOSE, 'EMISS', EMISS)
print('Gtr keys', list(Gtr.keys()), Gtr['gf'].shape, 'Gtu', Gtu['gf'].shape)
print(RR.head(3).to_string())
print(f3.head(3).to_string())
print(DEV.iloc[:2, :].to_string())
print('ids_states', ids_states[:5], 'f3_ids', f3_ids[:5])
```