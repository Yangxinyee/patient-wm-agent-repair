The Input section shows that drug bundles go through a DeepSets encoder followed by a layer norm (`encode_action.norm`). The END atom (kind 2) of the previous hour's infusion falls in the same minute as the next start, so a drug bundle usually holds two action atoms.

My earlier probe ("end->u attrs") was testing whether END atoms carry the rate and dose. That determines u_k, and therefore any counterfactual state I compute.

The rest of the doc, the heads section and "How a dose edit is read", should say exactly which atoms the dose edit modifies and how the heads read the state. That is what I need to reproduce the evaluation's counterfactual. I will print lines 54 to the end, about 2.3k chars.

```python
print('\n'.join(L_[54:])[:2700])
```