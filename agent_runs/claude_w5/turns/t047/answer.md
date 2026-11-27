The decoding shows the original model's state does not carry the information a counterfactual needs:
- The last glucose is decodable almost perfectly (R² 0.98).
- Raw doses at lags 1–3 reach R² of only 0.55–0.7, and mostly because doses follow glucose.
- The residual doses, the exogenous part that shows a causal effect, are nearly absent from `out` and `h` (R² about 0.03). Only the z posterior holds a little.

So a head-only readout cannot work. I need to build a dose memory in the recurrent core and test it locally, which means re-implementing the forward pass and checking it against the saved states. Next I'll log this finding in the notebook and look at the structure of my rebuilt bundles and of VOCAB. Short output only.

```python
with open('/work/notes.md','a') as fh:
    fh.write("\n## Runs 38-42: decoding of doses from original states (fwd_2, 100 train stays, ridge CV by stay)\n"
             "- g_last R2 0.98; ins lag1-3 R2 0.71/0.63/0.57; ne lag1-3 0.60/0.56/0.54 (mostly via glucose/MAP).\n"
             "- RESIDUAL doses (exogenous part): insres1 R2 0.25 (all), 0.03 (out/h only); neres1 0.15 / ~0.\n"
             "=> state does not carry dose memory; head-only readout insufficient. Need core-level memory injection.\n"
             "- Principle: edited prediction = m0 + sum_j beta_j*(dose - policy_mean(g)) (g-formula; centering keeps factual NLL).\n")
b0_=bundles_B[list(bundles_B.keys())[0]] if isinstance(bundles_B,dict) else bundles_B[0]
print(type(bundles_B).__name__, len(bundles_B), type(b0_).__name__, len(b0_))
print(repr(b0_[:3])[:800])
print(list(VOCAB.keys())); print(VOCAB.get('config'))
```