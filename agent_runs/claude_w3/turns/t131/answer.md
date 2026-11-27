The kernel still holds many objects from earlier work: encoder helpers such as `enc_bundle`, `atom_emb` and `build_bundles`, the policy arrays, and fwd_4 states. ARCHITECTURE.md is short (5.2k characters), so I'll read it in full again. Exact knowledge of how a bundle updates the state is needed before I design the dose-memory edit. I expect it to show where a dose enters (the rate/dose numeric fields), how h decays, and how the value head reads the state. That tells me which weights can carry a causal dose effect.

```python
print(arch)
```