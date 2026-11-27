Run 250 showed that the z-token channel is too weak:
- **Gradient size.** The next-reading prediction moves only about 0.1 mg/dL per unit change of the z token (finger), and the token's norm is about 12.7.
- **Native dose direction.** It is already nearly the most efficient direction for glucose: −0.33 finger per unit residual, close to the maximum possible. It has unwanted cross-effects, though: MAP moves +0.10 to +0.12 under insulin, and glucose moves −0.69 under norepinephrine.
- **Minimum-norm directions.** A clean direction (glucose only, no MAP change) would need a norm of 79 for insulin and 230 for norepinephrine per unit effect. That is far beyond the linear range, especially because the token is layer-normed before attention.

**Conclusion:** the residual is computed correctly, but it needs an amplifier further downstream. There are two candidates:
- a rank-one addition to the prelude value projection of the heads that attend u/z on drug bundles. It would read a token direction that normal tokens never use;
- the head-side z_prior → z_head path into the view.

To choose, I need the exact architecture of the prelude, the view and z_prior: what z_prior reads, and the head size and norms. Those details were lost in the restart. I'll print ARCHITECTURE.md and the shapes of the prelude, z_prior and z_head tensors.

**What would decide it:** if z_prior reads a state that already contains the drug-bundle signal, the head-side amplifier is simpler. Otherwise I'll build the prelude value-projection amplifier.

```python
print(arch)
print({n: SHAPES[n] for n in NAMES if 'prelude' in n or n.startswith('z_') or 'view' in n or 'query' in n})
```