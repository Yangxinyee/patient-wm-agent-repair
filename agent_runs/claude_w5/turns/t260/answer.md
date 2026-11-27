This narrows it down. At reading bundles a single local step matches the tool to 1e-6. At drug bundles the z posterior matches exactly (error 0.0), but every h slot is off by about 1e-3 relative (1.85 absolute per slot) and out by about 1e-3. Since z = z_post([ctx, e, u]) is exact, ctx, e and u are computed correctly. The discrepancy must be in how the prelude, blocks or coda use these inputs, and only when the action token is non-empty.

Candidates include a different token for the action in the prelude than in z_post (per-atom tokens, or an extra embedding), or memory or cross-attention entries that differ at drug bundles. I'll re-read the core section of ARCHITECTURE.md for anything about how the action enters the prelude and blocks that I may have missed.

```python
i0 = arch.find('## Core') if '## Core' in arch else arch.find('prelude')
print(len(arch), i0)
seg = arch[max(0,i0-2500): i0+6500]
print(seg)
```