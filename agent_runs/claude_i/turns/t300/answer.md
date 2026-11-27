That explains the contradiction:
- **Bias patch:** changing the z_in bias by +100 has no effect, so `core_step` never reads `WT['z_in.bias']`. My earlier "sensitivities" must come from something else.
- **z input:** shifting all of z by +2 moves the out vector by only 2e-5.

So in my reproduction, z barely reaches the output vector. That questions the whole z_post → z_in injection plan: if z hardly influences the state, the z_in column is a dead end, unless z acts mainly through h (the slots) rather than out.

Before spending more runs I need the authoritative description of how z enters the core step. I'll pull only the ARCHITECTURE.md lines that mention z_in, z_post, z_head, z_prior or the prelude. I'll also measure the effect of z+2 on h, not just on out.

```python
lines = open('/model/ARCHITECTURE.md').read().splitlines()
hits = [i for i,l in enumerate(lines) if any(s in l for s in ('z_in','z_post','z_head','z_prior','prelude','posterior'))]
for i in hits[:30]: print(i, lines[i][:220])
hb, _ = core_step(hf, EtW[k], UtW[k], z+2.0, EtW[idx_m], ages, {'incl':1,'unit':'s','age':'both'})
ha, _ = core_step(hf, EtW[k], UtW[k], z, EtW[idx_m], ages, {'incl':1,'unit':'s','age':'both'})
print('z+2 dh', float((hb-ha).norm()), '|h|', float(ha.norm()))
```