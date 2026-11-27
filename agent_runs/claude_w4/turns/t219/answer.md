Run 192 clarified v3's parts:
- **Readouts:** BB3 holds three readout vectors on the action token: insulin rate (tI), norepinephrine rate (fN) and insulin presence (pI).
- **Value directions:** aI3, aN3 and aP3 are unit directions in head 1's 48-dimensional value space.
- **Injection directions:** dI and dN are injection directions with norms 13 and 29.
- **Missing piece:** nothing in v3 can depend on the state. Its only stand-in for the expected dose is the presence term pI.

The v1 edits are dropped, so the model is back to the original.

My conclusion is that v3 alone would bias factual predictions by k·(E[rate|state] − constant), which is too risky for the 0.005-nat limit. I will build v4 in the prelude FFN instead. It uses gated unit pairs:
- **Unit A:** GELU(M·(gate−th) + c·x).
- **Unit B:** GELU(M·(gate−th)).
- **Output:** A − B, scaled to inject along a chosen direction.

The input x = rate − Ê[rate|state] is a single linear readout on the FFN input norm3(x1). It is fitted jointly across dose multipliers so that it tracks the edited rate but subtracts only the part predictable from the state.

This run checks the pieces I need:
- what space Gm lives in: is it the gradient of the four predicted means (finger, lab, arterial, cuff) per unit change of E, and what do Gm·dI and Gm·dN give?
- what the replica `prelude2` returns;
- which prelude FFN units are nearly dead at slot tokens and can be repurposed.

What I expect: a near-diagonal Gm·dI (finger and lab nonzero, MAP about zero) would confirm the min-norm direction. Units with tiny maximum contribution would serve as carriers.

```python
import torch
print("type load:", type(load('core.prelude.0.ffn.0.weight')))
GmA = np.asarray(Gm, float)
for nm_, v in [('dI', dI), ('dN', dN)]:
    print(nm_, "Gm@d =", np.round(GmA @ np.asarray(v, float), 3))
print("Gm row norms", np.round(np.linalg.norm(GmA, axis=1), 4), " cos(rows):", np.round(np.corrcoef(GmA)[0], 3))
Wp = {n: load_original(n) for n in NAMES if n.startswith('core.prelude')}
Wp = {k: (v if isinstance(v, torch.Tensor) else torch.tensor(v)) for k, v in Wp.items()}
out_d = prelude2(TOK[:200], Wp)
print("prelude2 keys:", {k: tuple(v.shape) for k, v in out_d.items()})
if 'E' in out_d:
    Eref = torch.as_tensor(np.asarray(E_ref_d[:200]), dtype=torch.float32)
    print("E vs E_ref_d max err", float((out_d['E'] - Eref).abs().max()), " vs E0:", float((out_d['E'] - torch.as_tensor(np.asarray(E0[:200]),dtype=torch.float32)).abs().max()))
o_all_d = prelude2(TOK, Wp); o_all_n = prelude2(TOKn, Wp)
hid_d = o_all_d['hid'][:, :4] if o_all_d['hid'].shape[1] == 7 else o_all_d['hid']
hid_n = o_all_n['hid'][:, :4] if o_all_n['hid'].shape[1] == 7 else o_all_n['hid']
a_d = torch.nn.functional.gelu(hid_d) if float(hid_d.min()) < -0.2 else hid_d
a_n = torch.nn.functional.gelu(hid_n) if float(hid_n.min()) < -0.2 else hid_n
w2n = Wp['core.prelude.0.ffn.2.weight'].norm(dim=0)
contrib = torch.cat([a_d.reshape(-1, 768), a_n.reshape(-1, 768)]).abs().max(0).values * w2n
order = torch.argsort(contrib)
print("hid shape", tuple(o_all_d['hid'].shape), "min hid", float(o_all_d['hid'].min()))
print("deadest units:", order[:12].tolist(), np.round(contrib[order[:12]].numpy(), 5))
print("median contrib", float(contrib.median()), " x2 norm median", float(o_all_d['x2'][:, :4].norm(dim=-1).median()) if 'x2' in o_all_d else '')
```