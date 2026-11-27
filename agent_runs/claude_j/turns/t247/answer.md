Run 219 refutes the input-amplification idea. Scaling the dose numerics by 4 leaves both the counterfactual glucose shift (about −0.86 under teacher forcing) and the factual NLL essentially unchanged. The model's dose signal comes through the bin embeddings and is squashed by the layer norms, so making the inputs bigger does not help.

That leaves the head readout. I now suspect its poor trade-off in the frontier comes from overfitting. A 192-feature ridge was fitted on about 80 stays per fold. Two fixes are possible: a low-dimensional readout restricted to the directions in which dose doubling actually moves the head features, or more stays.

Next I will inspect the stored counterfactual dataset (`M` and the `WC` features) to see how its rows match up. If its rows can be paired between counterfactual and factual by stay, event and minute, I can test the low-dimensional readout directly.

```python
print(type(M).__name__, M.shape, list(M.columns))
print(M['drug'].value_counts().to_dict(), sorted(M['m'].unique().tolist())[:8], sorted(M['h0'].unique().tolist()))
print(type(WC).__name__, tuple(WC.shape), type(TH).__name__, getattr(TH,'shape',None), type(LG).__name__, tuple(LG.shape))
print(M.head(2).to_string()[:400])
```