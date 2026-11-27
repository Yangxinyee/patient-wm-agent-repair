The search shows three key lines in ARCHITECTURE.md:
- line 43: "The state update uses the posterior mean";
- line 57: "At evaluation, z is drawn from the prior 4 times and the probabilities are averaged";
- line 83: from hour 13 on, the model draws its own values bundle by bundle and feeds them back.

So the posterior (z_post) affects the recurrent state update, and the prior affects the emission probabilities. In the rollout, a drug bundle's posterior is computed from the drug atoms plus the prior-sampled readings. That means v1's z_post edit should still have reached the state, unless the gating misfired or the rollout's drug bundles are encoded differently.

I need the exact rollout description (lines ~36–100) to see which networks see u_k in the rollout. This tells me where an injection actually reaches future readings. The file is only 5,234 characters, so I will print that section.

```python
print('\n'.join(l[:200] for l in lines[30:100]))
```