Run 280's grid found a good trade-off. With a small ridge (1e-4) and constraint weight 1e-4 the readouts are accurate where it matters and small elsewhere:

| | v6 (this fit) | v4 / target |
|---|---|---|
| Insulin channel error (rate − 1.77·presence) | 0.097 rate units | ≈ unconstrained fit |
| NE readout error | 0.052 | ≈ unconstrained fit |
| Readout on slot tokens | ≈ 1.2 | ~760 (v4) |
| Readout on reading tokens | ≈ 1.5 | ~3000 (v4) |
| Leak through head-1 attention | ≈ 0.002 rate units | ~±1 (v4) |

The run crashed in the optional channel check, so nothing was saved and the active edit is still the broken v4. The replica function `prelude_b` failed on a type error, apparently because one of its globals has been overwritten.

**Next:**
- Skip that check.
- Build and save v6 with this fit, gains 1 and 1.
- Print the sel100 stay list (the one with known truth) for the forward pass.

The forward pass that follows is the real test of v6:

| | What it would show |
|---|---|
| **Confirms the leak was the problem** | The rollout drift relative to the original disappears, and doubling insulin lowers glucose (with roughly linear x2/x½ responses). |
| **Refutes it** | Drift or wrong sign persists, pointing at the injection target itself. |

```python
import json
lam,wo=1e-4,1e-4
B=np.linalg.solve(Guu+wo*(Gs+Ge)+1e4*Gc+lam*np.eye(192),XuF)
kvn,Won,Wbn=build6(B,1,1)
for n,x in [('core.prelude.0.attn.kv.weight',kvn),('core.prelude.0.attn.out.weight',Won),('core.b_in.weight',Wbn)]:
    save_edit(n,torch.tensor(x,dtype=torch.float32))
np.savez('/work/v6_parts.npz',B=B,wo=wo,lam=lam)
print('saved v6(1,1):',edited(), 'kv |d| %.1f'%np.linalg.norm(kvn-kv0))
with open('/work/notes.md','a') as f:
    f.write('- Run 280 grid: lam=1e-4,wo=1e-4 -> drug err [.103 .052 .012], insComb .097, slot readout rms 1.2, e/t5 1.5 (v4: 760/3000).\n'
            '- Run 281: SAVED v6(1,1) = build6(B,1,1), B in /work/v6_parts.npz (prelude_b now crashes: some global clobbered; not needed).\n')
s=json.load(open('/work/sel100_both.json'))
st=s['stays'] if isinstance(s,dict) else s
print('n=',len(st)); print(json.dumps(st))
```