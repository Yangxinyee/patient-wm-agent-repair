Ridge decoders at the drug bundles show the dose innovation is only partly represented:
- **Insulin:** it decodes with R² of about 0.4 from z, `out` or h. The policy-expected dose decodes better from `out` (0.91) than the actual dose (0.78).
- **Norepinephrine:** the innovation decodes with R² of only 0.16–0.23.

So the model mostly tracks what the policy would do and only weakly tracks how the actual dose deviated. That fits the absorbed-policy finding.

Two plans are open:
- **Plan B:** a lag-1 innovation correction placed in the value head. This is a rank-one edit of its output layer: a logit tilt toward lower or higher bins, proportional to a linear readout of the head's hidden layer that estimates the innovation. I'd calibrate it with dose-edit passes.
- **Plan A:** a cross-attention memory of past doses through the END atoms in M. It could get the recovery right, but it needs an exact replica.

I'll start with B. To do that I need the value head's exact structure.

```python
heads=[n for n in NAMES if not (n.startswith('core.') or n.startswith('embed') or n.startswith('encode'))]
print('; '.join('%s%s'%(n,tuple(SHAPES[n])) for n in heads))
```