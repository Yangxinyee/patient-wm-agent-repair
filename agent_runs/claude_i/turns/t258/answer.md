The full core step is now clear. The prelude is pre-norm and E passes through `prelude_norm`, so any signal sent through the z token is bounded and nonlinear. That matches the earlier finding that z-dim Jacobians at drug bundles are tiny. A pure z-path injection may need large, out-of-distribution token changes.

Before committing to that, I'll read the rest of the heads section (lines 59 to 91). The value head gets extra inputs: a last-value numeric block, a transition table, and the action head's pending vector p. If any head input carries the running infusion directly, a head-side edit would avoid fighting the LayerNorms. The finding is confirmed if the value head reads something dose- or pending-related, and refuted if it only reads view plus last-value features.

```python
for i in range(59,92): print(i, A[i][:240])
```